import time

import deepspeed
import math
import pandas as pd
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from transformers import BertTokenizer, BertForSequenceClassification

import dist


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


def prepare_dataloader():
    dataset = MyDataset()

    trainset, validset = random_split(dataset, lengths=[0.9, 0.1], generator=torch.Generator().manual_seed(42))
    tokenizer = BertTokenizer.from_pretrained("/data3/ysh/huggingface/bert-base-chinese")

    def collate_func(batch):
        texts, labels = [], []
        for item in batch:
            texts.append(item[0])
            labels.append(item[1])
        inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        inputs["labels"] = torch.tensor(labels)
        return inputs

    sampler = DistributedSampler(validset, shuffle=False, drop_last=False) if dist.is_distributed() else None
    validloader = DataLoader(
        dataset=validset,
        batch_size=16,
        collate_fn=collate_func,
        shuffle=False,
        num_workers=4,
        sampler=sampler
    )

    return trainset, validloader


def prepare_model():
    model = BertForSequenceClassification.from_pretrained("/data3/ysh/huggingface/bert-base-chinese")

    lora_config = LoraConfig(target_modules=["query", "key", "value"])

    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    return model


batch_size = 32
grad_accumulation_steps = 2
epoch = 5


def evaluate(model, validloader, accelerator: Accelerator):
    model.eval()
    acc_num = 0
    with torch.no_grad():
        for batch in validloader:
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            pred, refs = accelerator.gather_for_metrics((pred, batch["labels"]))
            acc_num += (pred.long() == refs.long()).float().sum()
    return acc_num / len(validloader.dataset)


def initialize_deepspeed(model, trainset, total_num_steps):
    ds_config = {"train_micro_batch_size_per_gpu": batch_size,
                 "gradient_accumulation_steps": grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": 1e-5, "weight_decay": 0.0,
                                                           }},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": total_num_steps, "warmup_min_lr": 0,
                                          "warmup_max_lr": 1e-5, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "bf16": {"enabled": True},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, trainloader, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )

    model_engine.save_checkpoint()
    model_engine.load_checkpoint()

    return model_engine, optimizer, trainloader, scheduler


def train(model_engine, optimizer, trainloader, validloader, resume, epoch=3, log_step=10):
    global_step = 0
    start_time = time.time()

    resume_step = 0
    resume_epoch = 0

    if resume is not None:
        _, client_sd = model_engine.load_checkpoint(**resume)
        step = client_sd['step']
        steps_per_epoch = math.ceil(len(trainloader) / grad_accumulation_steps)
        resume_step = global_step = step
        resume_epoch = resume_step // steps_per_epoch
        resume_step -= resume_epoch * steps_per_epoch

    for ep in range(resume_epoch, epoch):
        if resume and ep == resume_epoch and resume_step != 0:
            import itertools
            active_dataloader = itertools.islice(iter(trainloader), resume_step, None)
        else:
            active_dataloader = trainloader

        for step, batch in enumerate(active_dataloader):
            for _ in range(grad_accumulation_steps):
                # forward() method
                loss = model_engine(batch).loss

                # runs backpropagation
                model_engine.backward(loss)

                # weight update
                model_engine.step()

            global_step += 1

            if global_step % log_step == 0:
                loss = accelerator.reduce(loss, "mean")
                d = torch.sum(batch["input_ids"])
                # accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}")
                accelerator.print(f"ep: {ep}, global_step: {global_step}, loss: {loss.item()}, data: {d}")
                accelerator.log({"loss": loss.item()}, global_step)

            if global_step % 50 == 0 and global_step != 0:
                client_sd['step'] = step
                ckpt_id = loss.item()
                model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)

            # save checkpoint
            if step % args.save_interval:
                client_sd['step'] = step
                ckpt_id = loss.item()
                model_engine.save_checkpoint(args.save_dir, ckpt_id, client_sd=client_sd)

        acc = evaluate(model, validloader, accelerator)
        accelerator.print(f"ep: {ep}, acc: {acc}, time: {time.time() - start_time}")
        accelerator.log({"acc": acc}, global_step)


def main():
    from accelerate import DeepSpeedPlugin

    accelerator = Accelerator(
        project_dir="ckpts",
        gradient_accumulation_steps=2,
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config="/data2/shyue/ysh/paper-code/lang-point/LMM-Grounder/configs/zero_3_stage.json",
        )
    )

    accelerator.init_trackers("runs")

    trainset, validloader = prepare_dataloader()

    num_steps_per_epoch = len(trainset) // batch_size // grad_accumulation_steps
    total_steps = num_steps_per_epoch * epoch

    model = prepare_model()
    model_engine, optimizer, trainloader, scheduler = initialize_deepspeed(model, trainset, total_steps)

    train(
        model_engine,
        optimizer, trainloader,
        validloader, epoch=epoch,
        resume={"load_dir": "output", "tag": "300"}
    )


if __name__ == "__main__":
    main()

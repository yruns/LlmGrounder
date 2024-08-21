import torch
from torchvision import transforms, datasets, models
import lightning as L


class LitSegmentation(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=21)
        self.loss_fn = torch.nn.CrossEntropyLoss()


    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)


class SegmentationData(L.LightningDataModule):
    def prepare_data(self):
        datasets.VOCSegmentation(root="data", download=True)

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        train_dataset = datasets.VOCSegmentation(root="data", transform=transform, target_transform=target_transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)


if __name__ == "__main__":
    model = LitSegmentation()
    data = SegmentationData()
    from lightning.pytorch.strategies import DeepSpeedStrategy

    deepspeed_config = {
        "zero_allow_untested_optimizer": True,
        "optimizer": {
            "type": "OneBitAdam",
            "params": {
                "lr": 3e-5,
                "betas": [0.998, 0.999],
                "eps": 1e-5,
                "weight_decay": 1e-9,
                "cuda_aware": True,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "last_batch_iteration": -1,
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            },
        },
        "zero_optimization": {
            "stage": 2,  # Enable Stage 2 ZeRO (Optimizer/Gradient state partitioning)
            "contiguous_gradients": True,  # Reduce gradient fragmentation.
            "overlap_comm": True,  # Overlap reduce/backward operation of gradients for speed.
            "allgather_bucket_size": 2e8,  # Number of elements to all gather at once.
            "reduce_bucket_size": 2e8,  # Number of elements we reduce/allreduce at once.
        },
    }

    trainer = L.Trainer(
        max_epochs=10,
        strategy=DeepSpeedStrategy(config=deepspeed_config)
    )
    trainer.fit(model, data)
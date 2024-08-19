import weakref
from typing import Union

from accelerate import Accelerator

from trim.callbacks.misc import *
from trim.utils.events import EventStorage


class TrainerBase(object):
    need_prepare_by_accelerator_objects = [
        "model", "train_loader", "val_loader", "optimizer", "lr_scheduler"
    ]

    def __init__(self):
        self.accelerator: Union[Accelerator, None] = None
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criteria = None
        self.train_loader = None
        self.val_loader = None
        self.scaler = None
        self.logger = None
        self.wandb = None
        self.callbacks = []
        self.stroage: EventStorage
        self.comm_info = dict()
        self.best_metric_value = float("-inf")
        self.output_dir = None

        self.start_epoch = 0
        self.max_epoch = None
        self.completed_steps = 0
        self.resume_step = 0
        self.total_train_steps = 0
        self.num_update_steps_per_epoch = 0
        self.data_iterator = None
        self.debug = False

    def training_setp(self, batch_data, batch_index):
        pass

    def test_step(self, batch_data, batch_index):
        pass

    def on_training_epoch_start(self):
        for callback in self.callbacks:
            callback.on_training_epoch_start()

    def on_training_epoch_end(self):
        for callback in self.callbacks:
            callback.on_training_epoch_end()
        self.storage.reset_histories()  # reset histories (required)
        self.wandb.save(os.path.join(self.output_dir, "train.log"))

    def on_training_setp_start(self):
        for callback in self.callbacks:
            callback.on_training_setp_start()

    def on_training_setp_end(self):
        for callback in self.callbacks:
            callback.on_training_step_end()

    def on_training_phase_start(self):
        for callback in self.callbacks:
            callback.on_training_phase_start()

    def on_training_phase_end(self):
        for callback in self.callbacks:
            callback.on_training_phase_end()

    def configure_callbacks(self):
        for callback in self.callbacks:
            assert isinstance(callback, CallbackBase)
            callback.trainer = weakref.proxy(self)
            callback.accelerator = weakref.proxy(self.accelerator)

    def configure_optimizers(self):
        pass

    def configure_model(self):
        pass

    def configure_dataloader(self):
        pass

    def configure_criteria(self):
        pass

    def configure_scaler(self):
        pass

    def configure_wandb(self):
        pass

    def setup(self):
        from trim.utils import comm
        comm.lazy_init_accelerate(self.accelerator)
        self.configure_wandb()
        self.configure_model()
        self.configure_dataloader()
        self.configure_optimizers()
        self.configure_criteria()
        self.configure_scaler()
        self.configure_callbacks()

        wait_prepare_obj_names = []
        wait_prepare_objs = []
        for obj_name in self.need_prepare_by_accelerator_objects:
            if hasattr(self, obj_name):
                wait_prepare_obj_names.append(obj_name)
                wait_prepare_objs.append(getattr(self, obj_name))

        prepared_objs = self.accelerator.prepare(*wait_prepare_objs)
        for obj_name, prepared_objs in zip(wait_prepare_obj_names, prepared_objs):
            setattr(self, obj_name, prepared_objs)

    def fit(self):
        self.setup()
        with EventStorage() as self.storage:
            self.on_training_phase_start()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.model.train()
                if self.debug:
                    from itertools import islice
                    self.data_iterator = enumerate(islice(self.train_loader, 10))
                else:
                    self.data_iterator = enumerate(self.train_loader)
                if self.accelerator.sync_gradients:
                    self.on_training_epoch_start()
                # => run_epoch
                for batch_index, batch_data in self.data_iterator:
                    self.comm_info["iter"] = batch_index
                    if self.accelerator.sync_gradients:
                        self.on_training_setp_start()
                    self.training_setp(batch_data, batch_index)
                    if self.accelerator.sync_gradients:
                        self.on_training_setp_end()
                # => after epoch
                if self.accelerator.sync_gradients:
                    self.on_training_epoch_end()
            # => after train
            self.on_training_phase_end()

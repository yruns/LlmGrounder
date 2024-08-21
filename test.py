from typing import Optional, Union, Dict, Any, List

import torch
from torch import nn


from transformers import Trainer

class MyTrainer(Trainer):

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.is_local_process_zero()
        return super().training_step(model, inputs)


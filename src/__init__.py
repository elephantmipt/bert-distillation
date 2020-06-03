from catalyst.contrib.dl.callbacks.wandb import WandbLogger
from catalyst.dl import registry
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss, KLDivLoss, MSELoss
from torch_optimizer import Ranger

from .callbacks import (
    CosineLossCallback,
    KLDivLossCallback,
    MaskedLanguageModelCallback,
    MSELossCallback,
)
from .experiment import Experiment  # noqa: F401
from .models import BertForMLM, DistilbertStudentModel
from .runners import DistilMLMRunner as Runner  # noqa: F401

registry.Model(BertForMLM)
registry.Model(DistilbertStudentModel)

registry.Optimizer(Ranger)

registry.Callback(WandbLogger)

registry.Criterion(CosineEmbeddingLoss)
registry.Criterion(CrossEntropyLoss)
registry.Criterion(KLDivLoss)
registry.Criterion(MSELoss)

registry.Callback(CosineLossCallback)
registry.Callback(MaskedLanguageModelCallback)
registry.Callback(KLDivLossCallback)
registry.Callback(MSELossCallback)

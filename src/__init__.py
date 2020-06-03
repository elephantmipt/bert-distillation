from catalyst.contrib.dl.callbacks.wandb import WandbLogger
from catalyst.dl import registry
from torch.nn import CosineEmbeddingLoss, CrossEntropyLoss
from torch_optimizer import Ranger

from .callbacks import CosineLossCallback, MaskedLanguageModelCallback
from .experiment import Experiment  # noqa: F401
from .models import BertForMLM, DistilbertStudentModel
from .runners import DistilMLMRunner as Runner  # noqa: F401

registry.Model(BertForMLM)
registry.Model(DistilbertStudentModel)

registry.Optimizer(Ranger)

registry.Callback(WandbLogger)

registry.Criterion(CosineEmbeddingLoss)
registry.Criterion(CrossEntropyLoss)

registry.Callback(CosineLossCallback)
registry.Callback(MaskedLanguageModelCallback)

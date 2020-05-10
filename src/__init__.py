from catalyst.contrib.dl.callbacks.wandb import WandbLogger
from catalyst.dl import registry
from torch_optimizer import Ranger

from .experiment import Experiment  # noqa: F401
from .models import BertForMLM, DistilbertStudentModel
from .runners import DistilMLMRunner as Runner  # noqa: F401

registry.Model(BertForMLM)
registry.Model(DistilbertStudentModel)

registry.Optimizer(Ranger)

registry.Callback(WandbLogger)

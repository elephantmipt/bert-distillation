from catalyst.dl import registry
from torch_optimizer import Ranger

from .callbacks import (
    CosineLossCallback,
    KLDivLossCallback,
    MaskedLanguageModelCallback,
    MSELossCallback,
    PerplexityMetricCallbackDistillation,
)
from .experiment import Experiment  # noqa: F401
from .models import BertForMLM, DistilbertStudentModel
from .runners import DistilMLMRunner as Runner  # noqa: F401

registry.Model(BertForMLM)
registry.Model(DistilbertStudentModel)

registry.Optimizer(Ranger)

registry.Callback(CosineLossCallback)
registry.Callback(MaskedLanguageModelCallback)
registry.Callback(KLDivLossCallback)
registry.Callback(MSELossCallback)
registry.Callback(PerplexityMetricCallbackDistillation)

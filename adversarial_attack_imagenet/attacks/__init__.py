from .base import Attack  # noqa: F401

# FixedEpsilonAttack subclasses
from .projected_gradient_descent import (  # noqa: F401
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
)

from .fast_gradient_method import (  # noqa: F401
    L1FastGradientAttack,
    L2FastGradientAttack,
    LinfFastGradientAttack,
)

from .deepfool import L2DeepFoolAttack, LinfDeepFoolAttack

FGM = L2FastGradientAttack
FGSM = LinfFastGradientAttack
L1PGD = L1ProjectedGradientDescentAttack
L2PGD = L2ProjectedGradientDescentAttack
LinfPGD = LinfProjectedGradientDescentAttack
PGD = LinfPGD


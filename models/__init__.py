from .losses import ASoftmax, CosFace, MEMS
from .networks import networks

losses = {
    "a-softmax": ASoftmax,
    "cosface": CosFace,
    "lmcl": CosFace,
    "mems": MEMS
}
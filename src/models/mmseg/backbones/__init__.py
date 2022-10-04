from .mit import MixVisionTransformer
from .pvt import PyramidVisionTransformerV2
from .coat import CoaT
from .dvt import DualViT
from .swin_transformer import *
BACKBONES = {_.__name__: _ for _ in [
    MixVisionTransformer,
    PyramidVisionTransformerV2,
    CoaT,
    DualViT,
    SwinTransformer,
]}
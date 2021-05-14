import torch.nn as nn
import torch

from mmedit.models.builder import build_component
from mmedit.models.registry import BACKBONES
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from ... import builder

from mmcv.runner import auto_fp16, load_checkpoint
from mmedit.utils import get_root_logger


@BACKBONES.register_module()
class SwinuperEncoderDecoder(nn.Module):
    """Swin+Upernet Encoder-Decoder used in Global&Local model.

    This implementation follows:
    Globally and locally Consistent Image Completion

    The architecture of the encoder-decoder is:\
        (conv2d x 6) --> (dilated conv2d x 4) --> (conv2d or deconv2d x 7)

    Args:
        encoder (dict): Config dict to encoder.
        decoder (dict): Config dict to build decoder.
        dilation_neck (dict): Config dict to build dilation neck.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 pretrained=None):
        super().__init__()
        self.backbone = builder.build_backbone(backbone)
        self._init_decode_head(decode_head)

        self.init_weights(pretrained=pretrained)

        assert hasattr(self, 'decode_head') and self.decode_head is not None


    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    @auto_fp16()
    def forward(self, img):
        """Forward Function.

        Args:
            img (torch.Tensor): Input tensor with shape of (n, c+1, h, w).
                                Last channel is mask.

        Returns:
            torch.Tensor: Output tensor with shape of (n, c, h', w').
        """
        x = self.extract_feat(img[:, :-1])
        out = self.decode_head(x)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out


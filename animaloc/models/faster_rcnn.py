import torch.nn as nn

from typing import Optional

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone,  _validate_trainable_layers
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from .register import MODELS


def _fasterrcnn_resnet_fpn(
    architecture: str,
    num_classes: int, 
    pretrained_backbone: bool = True, 
    trainable_backbone_layers: Optional[int] = None, 
    anchor_sizes: Optional[tuple] = None,
    **kwargs
    ) -> FasterRCNN:

    assert architecture in ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2']

    trainable_backbone_layers = _validate_trainable_layers(pretrained_backbone, trainable_backbone_layers, 5, 3)

    backbone = resnet_fpn_backbone(architecture, pretrained_backbone, trainable_layers=trainable_backbone_layers)
    if anchor_sizes is not None:
        anchor_sizes = tuple([tuple([i,]) for i in anchor_sizes])
        aspects = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        anchor_sizes = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspects)

    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchor_sizes, **kwargs)

    return model

@MODELS.register()
class FasterRCNNResNetFPN(nn.Module):
    '''
    Build a Faster R-CNN model with a ResNet-FPN backbone. The ResNet architecture have to be specified in
    'architecture' argument. 

    This class was inspired by the ``fasterrcnn_resnet50_fpn`` function of torchvision, for details
    see https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L299.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Args:
        architecture (str): resnet architecture. Possible values are 'ResNet', 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool, optional): If True, returns a model with backbone pre-trained on Imagenet. 
            Defautls to True
        trainable_backbone_layers (int, optional): number of trainable (not frozen) resnet layers starting from 
            final block. Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
            Defaults to None
        **kwargs: additional FasterRCNN arguments
    '''

    def __init__(
        self,
        architecture: str,
        num_classes: int, 
        pretrained_backbone: bool = True, 
        trainable_backbone_layers: Optional[int] = None, 
        anchor_sizes: Optional[tuple] = None,
        **kwargs
        ) -> None:
        super().__init__()

        self.arch = _fasterrcnn_resnet_fpn(
            architecture=architecture, num_classes=num_classes, 
            pretrained_backbone=pretrained_backbone, 
            trainable_backbone_layers=trainable_backbone_layers, 
            anchor_sizes=anchor_sizes,
            **kwargs)
    
    def forward(self, *args, **kwargs):
        return self.arch(*args, **kwargs)
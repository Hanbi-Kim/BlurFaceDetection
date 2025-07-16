from .simple_cnn import SimpleCNN
from .shallow_cnn import ShallowCNN
from .vgg16 import VGG16
from .resnet18_transfer import ResNet18Transfer
from .resnet18_4ch import ResNet18_4ch   
from .resnet18_ch4_cbam import ResNet18_4ch_CBAM
from .resnet18_edgeattn import ResNet18_EdgeAttention
from .edge_resnet import EdgeResNet18

__all__ = ['SimpleCNN', 'ShallowCNN', 'VGG16', 'ResNet18Transfer','ResNet18_4ch','ResNet18_4ch_CBAM','ResNet18_EdgeAttention','ResNet18_4ch_CBAM','EdgeResNet18']

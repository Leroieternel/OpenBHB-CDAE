from .estimators import AgeEstimator, SiteEstimator
from .resnet3d import SupConResNet, SupRegResNet, SupCEResNet, LinearRegressor
from .alexnet3d import SupConAlexNet, SupRegAlexNet
from .densenet3d import SupConDenseNet, SupRegDenseNet
from .unet import UNet
from .unet_encoder import UNet_Encoder
from .unet_decoder import UNet_Decoder
from .unet_encoder_balanced import UNet_Encoder_b
from .unet_decoder_balanced import UNet_Decoder_b
from .unet_encoder_wo_sc import UNet_Encoder_1
from .unet_decoder_wo_sc import UNet_Decoder_1
from .unet_en_wo_sc import UNet_Encoder_b_wosc
from .unet_de_wo_sc import UNet_Decoder_b_wosc
from .alexnet import AlexNet2D_Dropout
from .alex3d_cnn import AlexNet3D_Dropout
from .alex3d_encoder import AlexNet3D_Encoder
from .alex3d_decoder import AlexNet3D_Decoder
from .alex3d_encoder_test import AlexNet3D_Encoder_1
from .alex3d_decoder_test import AlexNet3D_Decoder_1
# from wi_net import wi_net

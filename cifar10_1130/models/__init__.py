# from .vavanilla_resnet_cifar import vanilla_resnet20
from .vanilla_models.vanilla_resnet_cifar import vanilla_resnet8, vanilla_resnet14, vanilla_resnet20, vanilla_resnet32, vanilla_resnet44, vanilla_resnet56
from .noisy_resnet_cifar import noise_resnet8, noise_resnet14, noise_resnet20, noise_resnet32, noise_resnet44, noise_resnet56
from .fixnoisy_resnet_cifar import fixnoise_resnet8, fixnoise_resnet14, fixnoise_resnet20, fixnoise_resnet32, fixnoise_resnet44, fixnoise_resnet56
from .dropout_resnet_cifar import dropout_resnet8, dropout_resnet14, dropout_resnet20, dropout_resnet32, dropout_resnet44, dropout_resnet56
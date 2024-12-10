from .conv_ae import get_convae
from .backbones import get_pdn_medium, get_pdn_small

def get_autoencoder(model_name, **kwargs):
    if 'conv_ae' in model_name:
        return get_convae(**kwargs)
    else:
        raise ValueError(f"Invalid autoencoder model: {model_name}")

def get_backbone(model_name, **kwargs):
    if 'pdn_small' in model_name:
        return get_pdn_small(**kwargs)
    elif 'pdn_medium' in model_name:
        return get_pdn_medium(**kwargs)
    else:
        raise ValueError(f"Invalid backbone model: {model_name}")
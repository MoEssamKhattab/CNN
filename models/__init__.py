from .ResNetA import ResNetA
from .plainNet import plainNet

def build_model(model_name,config):
    if model_name =='resnet':
        model = ResNetA(config)
    elif model_name == 'plain_net': 
        model = plainNet(config)
    
    return model
    
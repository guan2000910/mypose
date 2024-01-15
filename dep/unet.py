# Initially from https://github.com/usuyama/pytorch-unet (MIT License)
# Architecture slightly changed (removed some expensive high-res convolutions)
# and extended to allow multiple decoders
from pyexpat import features
import torch
from torch import nn
import torchvision
import torch.nn.functional as F
import numpy as np



def myGELU(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(),
    )

def convreclu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.GELU(approximate='none'),
    )

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

# class ResNetUNet1(nn.Module):
#     def __init__(self, n_class, feat_preultimate=64, n_decoders=1):
#         super().__init__()

#        #  shared encoder
#         self.base_model = torchvision.models.resnet18(pretrained=True)
#         self.anmodel = torchvision.models.convnext_base(pretrained=True).features

#  #Print the named children of ConvNeXt's features
#         for name, child_module in self.anmodel.named_children():
#             print(f"Child Module: {name}")
#             print(child_module)

# 创建 ResNetUNet 模型的实例
#resnet_unet_model = ResNetUNet1(n_class=10)


class ResNetUNet(nn.Module):

    def __init__(self, n_class, feat_preultimate=64, n_decoders=1):
        super().__init__()

        #  shared encoder
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.ctmodel = torchvision.models.convnext_base(pretrained=True).features
        self.ct_layers = list(self.ctmodel.children())
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4),基础块
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)，基础块
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)，基础块
        simam_module()
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)，基础块

       #加入convnext：
        self.ctlayer0 = nn.Sequential(self.ct_layers[0]) #size=(N, 128, x.H/4, x.W/4)
        self.ctlayer1 = self.ct_layers[1]  #CNBlocks
        self.ctlayer2 = nn.Sequential(self.ct_layers[2]) #size=(N, 256, x.H/8, x.W/8)
        self.ctlayer3 = self.ct_layers[3]  #CNBlocks
        self.ctlayer4 = nn.Sequential(self.ct_layers[4])  #size=(N, 512, x.H/16, x.W/16)
        #self.ctlayer5 = self.ct_layers[5]  #CNBlocks
        #self.ctlayer6 = nn.Sequential(self.ct_layers[6]) #size=(N,1024,H/32,W/32) Sequential
         
 
        #  n_decoders
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.decoders = [dict(
            layer0_1x1=convrelu(64, 64, 1, 0),
            layer1_1x1=convrelu(64, 64, 1, 0),
            layer2_1x1=convrelu(128, 128, 1, 0),
            layer3_1x1=convrelu(256, 256, 1, 0),
            layer4_1x1=convrelu(512, 512, 1, 0),

            layerc1_1x1=convrelu(128, 128, 1, 0),
            layerc3_1x1=convrelu(256, 256, 1, 0),
            layerc5_1x1=convrelu(512, 512, 1, 0),
            #layerc6_1x1=convrelu(1024, 1024, 1, 0),

            #conv_upc5=convreclu(512 + 1024, 1024, 3, 1),
            conv_upc3=convreclu(256 + 512, 256, 3, 1),
            conv_upc1=convreclu(128 + 256, 128, 3, 1),

            conv_up3=convrelu(256 + 512, 512, 3, 1),
            conv_up2=convrelu(128 + 512, 256, 3, 1),
            conv_up1=convrelu(64 + 256, 256, 3, 1),
            conv_up0=convrelu(64 + 256, 128, 3, 1),
            conv_original_size=convrelu(256, feat_preultimate, 3, 1),
            conv_last=nn.Conv2d(feat_preultimate, n_class, 1),
        ) for _ in range(n_decoders)]
 
        # register decoder modules
        for i, decoder in enumerate(self.decoders):
            for key, val in decoder.items():
                setattr(self, f'decoder{i}_{key}', val)

    def forward(self, input, decoder_idx=None):
        if decoder_idx is None:
            assert len(self.decoders) == 1
            decoder_idx = [0]
        else:
            assert len(decoder_idx) == 1 or len(decoder_idx) == len(input)

        # encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        
        layerct0 = self.ctlayer0(input)
        layerct1 = self.ctlayer1(layerct0) #CNBlocks
        layerct2 = self.ctlayer2(layerct1) 
        layerct3 = self.ctlayer3(layerct2) #CNBlocks
        layerct4 = self.ctlayer4(layerct3)
        
        
        
        #layerct5 = F.interpolate(layerct4, scale_factor=0.5, mode='bilinear', align_corners=False)
        
        layers = [layer0, layer1, layer2, layer3, layer4]
        layercts = [layerct0, layerct1, layerct2, layerct3, layerct4]
        
        #layer4_fused = 0.9 * layer4 + 0.1 * layerct5

        # decoders
        out = []
        for i, dec_idx in enumerate(decoder_idx):
            decoder = self.decoders[dec_idx]
            batch_slice = slice(None) if len(decoder_idx) == 1 else slice(i, i + 1)
            x = decoder['layer4_1x1'](layer4[batch_slice])
            simam_module(x)
            x = self.upsample(x)

            x1 = decoder['layerc5_1x1'](layerct4[batch_slice])
            simam_module(x1)
            x1 = self.upsample(x1)
            for layer_idx in 3, 2, 1, 0:
                layer_slice = layers[layer_idx][batch_slice]
                #layer_ct_slice = layercts[layer_idx][batch_slice]  # 使用对应的 layerct 特征图
                layer_projection = decoder[f'layer{layer_idx}_1x1'](layer_slice)
                x = torch.cat([x, layer_projection], dim=1)
                x = decoder[f'conv_up{layer_idx}'](x)
                x = self.upsample(x)

           # x = decoder['conv_original_size'](x)
           # out.append(decoder['conv_last'](x))

            for layer_idx in 3, 1:
                layer_ct_slice = layercts[layer_idx][batch_slice]  # 使用对应的 layerct 特征图
                layerct_projection = decoder[f'layerc{layer_idx}_1x1'](layer_ct_slice) #得到投影特征
                x1 = torch.cat([x1, layerct_projection], dim=1) #拼接
                x1 = decoder[f'conv_upc{layer_idx}'](x1)
                x1 = self.upsample(x1)

            x1 = self.upsample(x1)
            x1 = torch.cat([x1, x], dim=1)
            x1 = decoder['conv_original_size'](x1)
            out.append(decoder['conv_last'](x1))

            

        if len(decoder_idx) == 1:
            #  out: 1 x (B, C, H, W)
            return out[0]
        else:
            #  out: B x (1, C, H, W)
            return torch.stack(out)[:, 0]

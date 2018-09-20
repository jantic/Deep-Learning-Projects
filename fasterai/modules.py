from fastai.torch_imports import *
from fastai.conv_learner import *


class ConvBlock(nn.Module):
    def __init__(self, ni, no, ks=3, stride=1, bn=True, pad=None, actn=True):
        super().__init__()   
        if pad is None: pad = ks//2//stride

        layers = [nn.Conv2d(ni, no, ks, stride, padding=pad)]
        if actn:
            layers.append(nn.LeakyReLU())
        if bn:
            layers.append(nn.BatchNorm2d(no))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class UpSampleBlock(nn.Module):
    @staticmethod
    def _conv(ni: int, nf: int):
        kernel_size = 3
        layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size//2, stride=1)]
        return nn.Sequential(*layers)

    @staticmethod
    def _icnr(x:torch.Tensor, scale:int =2, init=nn.init.kaiming_normal):
        new_shape = [int(x.shape[0] / (scale ** 2))] + list(x.shape[1:])
        subkernel = torch.zeros(new_shape)
        subkernel = init(subkernel)
        subkernel = subkernel.transpose(0, 1)
        subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                                subkernel.shape[1], -1)
        kernel = subkernel.repeat(1, 1, scale ** 2)
        transposed_shape = [x.shape[1]] + [x.shape[0]] + list(x.shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def __init__(self, ni: int, nf: int, scale=2):
        super().__init__()
        layers = []
        
        for i in range(int(math.log(scale,2))):
            layers += [UpSampleBlock._conv(ni, nf*4), 
                       nn.PixelShuffle(2)]
                       
        self.sequence = nn.Sequential(*layers)
        self._icnr_init()
        
    def _icnr_init(self):
        conv_shuffle = self.sequence[0][0]
        kernel = UpSampleBlock._icnr(conv_shuffle.weight)
        conv_shuffle.weight.data.copy_(kernel)
    
    def forward(self, x):
        return self.sequence(x)

class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.m = nn.Sequential(*layers)

    def forward(self, x): 
        return x + self.m(x) * self.res_scale


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out//2
        self.x_conv  = nn.Conv2d(x_in,  x_out,  1)
        self.tr_conv = UpSampleBlock(up_in, up_out, 2)
        self.bn = nn.BatchNorm2d(n_out)
        
    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p,x_p], dim=1)
        return self.bn(F.relu(cat_p))


def get_pretrained_resnet_base(layers_cut:int= 0):
    f = resnet34
    cut,lr_cut = model_meta[f]
    cut-=layers_cut
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers), lr_cut


class SaveFeatures():
    features=None
    def __init__(self, m): 
        self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): 
        self.features = output
    def remove(self): 
        self.hook.remove()
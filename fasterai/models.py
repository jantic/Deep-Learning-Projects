from fastai.torch_imports import *


class UpSampleBlock(nn.Module):
    @staticmethod
    def _conv(ni: int, nf: int, kernel_size:int=3, actn:bool=False, stride:int=1, normalizer=None):
        layers = [nn.Conv2d(ni, nf, kernel_size, padding=kernel_size//2, stride=stride)]
        if normalizer is not None: 
            layers.append(normalizer)
        if actn: 
            layers.append(nn.LeakyReLU())
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

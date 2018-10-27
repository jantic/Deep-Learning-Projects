from fasterai.modules import *
from abc import ABC, abstractmethod

class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def set_trainable(self, trainable: bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self, precompute: bool = False)->[]:
        pass

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)


class EDSRImageModifier(GeneratorModule):
    def __init__(self, nf=128):
        super().__init__() 
        rn, lr_cut = get_pretrained_resnet_base()

        self.rn = rn
        #set_trainable(rn, False)
        self.lr_cut = lr_cut
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6,7]]

        self.up0 = nn.Sequential(
            UpSampleBlock(512, 256, 2),
            UpSampleBlock(256, 128, 2),
            UpSampleBlock(128, 64, 2), 
            UpSampleBlock(64, 32, 2), 
            UpSampleBlock(32, 16, 2))
        self.up1 = nn.Sequential(
            UpSampleBlock(256, 128, 2),
            UpSampleBlock(128, 64, 2), 
            UpSampleBlock(64, 32, 2), 
            UpSampleBlock(32, 16, 2))
        self.up2 = nn.Sequential(
            UpSampleBlock(128, 64, 2),
            UpSampleBlock(64, 32, 2),
            UpSampleBlock(32, 16, 2))
        self.up3 = nn.Sequential(
            UpSampleBlock(64, 32, 2),
            UpSampleBlock(32, 16, 2))
        self.up4 = UpSampleBlock(64, 32, 2)   #64 in  
        nf_up=99
 
        mid_layers = [nn.BatchNorm2d(nf_up)]
        mid_layers += [ConvBlock(nf_up,nf, actn=False, bn=False)]
        
        for i in range(10): 
            mid_layers.append(
                ResSequential(
                [ConvBlock(nf, nf, actn=True, bn=False), 
                 ConvBlock(nf, nf, actn=False, bn=False)], 0.1)
            )
            
        mid_layers += [nn.BatchNorm2d(nf)]
        self.upconv = nn.Sequential(*mid_layers)
             
        out_layers = []
        out_layers += [ConvBlock(nf, 3, ks=3, actn=False, bn=False)]
        out_layers += [nn.Tanh()]
        self.out = nn.Sequential(*out_layers)

    def get_layer_groups(self, precompute: bool = False)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]
        
    def forward(self, x: torch.Tensor): 
        self.rn(x)
        x0 = self.up0(self.sfs[4].features)
        x1 = self.up1(self.sfs[3].features)
        x2 = self.up2(self.sfs[2].features)
        x3 = self.up3(self.sfs[1].features)
        x4 = self.up4(self.sfs[0].features) 
        x5 = self.upconv(torch.cat([x, x0, x1, x2, x3, x4], dim=1))
        return self.out(x5)    

class MinimalEDSRImageModifier(GeneratorModule):
    def __init__(self, nf=128):
        super().__init__() 

        layers = []
        layers += [ConvBlock(3,nf, actn=False)]
        
        for i in range(10): 
            layers.append(
                ResSequential(
                [ConvBlock(nf, nf, actn=True, bn=False), 
                 ConvBlock(nf, nf, actn=False, bn=False)], 0.1)
            )
            
        layers += [nn.BatchNorm2d(nf)]
        layers += [ConvBlock(nf, 3, bn=False, actn=False)]
        layers += [nn.Tanh()]
        self.out = nn.Sequential(*layers)

    def get_layer_groups(self, precompute: bool = False)->[]:
        return children(self)
        
    def forward(self, x): 
        return self.out(x)

class DCGenerator(GeneratorModule):
    def __init__(self, x_noise:int=64, nf:int=64, scale:int=64, sn=False, self_attention=False):
        super().__init__()
        cngf = nf//2
        for i in range(int(math.log(scale,2))):
            cngf*=2

        layers = [DeconvBlock(x_noise, cngf, 4, 1, 0, sn=sn)]
        scale_count = 0
        cndf = cngf
        num_layers = int(math.log(scale,2))-3
        
        for i in range(num_layers):
            use_attention = (i == num_layers-2) and self_attention
            layers.append(DeconvBlock(cngf, cngf//2, 4, 2, 1, sn=sn, self_attention=use_attention))
            cngf //= 2

        out = nn.ConvTranspose2d(cngf, 3, 4, 2, 1, bias=False)
        if sn:
            out = spectral_norm(out)
        layers.append(out)
        self.features = nn.Sequential(*layers)
        self.out = nn.Tanh()

    def get_layer_groups(self, precompute: bool = False)->[]:
        return [children(self)]

    def forward(self, input): 
        return self.out(self.features(input))
 
class Unet34(GeneratorModule): 
    def __init__(self, nf_factor:int=1, bn=True, sn=True, self_attention=False, leakyReLu=True , scale:int=1):
        super().__init__()
        assert (math.log(scale,2)).is_integer()
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]

        self.up1 = UnetBlock(512,256,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn)
        self.up2 = UnetBlock(256*nf_factor,128,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn)
        self.up3 = UnetBlock(256*nf_factor,64,256*nf_factor, sn=sn, self_attention=self_attention, leakyReLu=leakyReLu, bn=bn)
        self.up4 = UnetBlock(256*nf_factor,64,256*nf_factor, sn=sn, leakyReLu=leakyReLu, bn=bn)
        self.up5 = UpSampleBlock(256*nf_factor, 256*nf_factor, 2*scale, sn=sn, leakyReLu=leakyReLu, bn=bn) 
        self.out= nn.Sequential(ConvBlock(256*nf_factor, 3, ks=3, actn=False, bn=False, sn=sn), nn.Tanh())

    #Gets around irritating inconsistent halving come from resnet
    def _pad_xtensor(self, x, target):
        h = x.shape[2] 
        w = x.shape[3]

        target_h = target.shape[2]*2
        target_w = target.shape[3]*2

        if h<target_h or w<target_w:
            target = Variable(torch.zeros(x.shape[0], x.shape[1], target_h, target_w))
            target[:,:,:h,:w]=x
            return to_gpu(target)

        return x
           
    def forward(self, x_in: torch.Tensor):
        x = F.relu(self.rn(x_in))
        x = self.up1(x, self._pad_xtensor(self.sfs[3].features, x))
        x = self.up2(x, self._pad_xtensor(self.sfs[2].features, x))
        x = self.up3(x, self._pad_xtensor(self.sfs[1].features, x))
        x = self.up4(x, self._pad_xtensor(self.sfs[0].features, x))
        x = self.up5(x)
        x = self.out(x)
        return x
    
    def get_layer_groups(self, precompute: bool = False)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]
    
    def close(self):
        for sf in self.sfs: 
            sf.remove()

class LearnerGenModuleWrapper():
    def __init__(self, model: GeneratorModule, name:str):
        self.model = to_gpu(model)
        self.name = name

    def get_layer_groups(self, precompute):
        return self.model.get_layer_groups()
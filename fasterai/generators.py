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


class Unet34(GeneratorModule): 
    def __init__(self, nf_factor:int=1):
        super().__init__()
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]

        self.up1 = UnetBlock(512,256,512*nf_factor)
        self.up2 = UnetBlock(512*nf_factor,128,256*nf_factor)
        self.up3 = UnetBlock(256*nf_factor,64,128*nf_factor)
        self.up4 = UnetBlock(128*nf_factor,64,64*nf_factor)
        self.up5 = UpSampleBlock(64*nf_factor, 32*nf_factor, 2)    
        self.out= nn.Sequential(ConvBlock(32*nf_factor, 3, ks=3, actn=False, bn=False), nn.Tanh())

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
        x = F.leaky_relu(self.rn(x_in))
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
from fasterai.modules import *
from abc import ABC, abstractmethod

class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def set_trainable(self, trainable: bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self)->[]:
        pass

    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)

class ResnetImageModifier(GeneratorModule):     
    def __init__(self, nf:int=128):
        super().__init__() 
        
        self.rn, self.lr_cut = get_pretrained_resnet_base(1)
        
        self.color = nn.Sequential(
            ResBlock(256),
            ConvBlock(256, nf),
            ResBlock(nf),
            ConvBlock(nf, nf//4),
            ResBlock(nf//4),
            ConvBlock(nf//4, nf//16),
            ResBlock(nf//16),
            UpSampleBlock(nf//16, nf//16, 16),
            ConvBlock(nf//16,3, actn=False, bn=False)
        )
        
        self.out = nn.Sequential(
            ConvBlock(6,3, actn=False, bn=False)
        )
    

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]
        
    def forward(self, orig): 
        x = self.rn(orig)
        x = self.color(x)
        return F.tanh(self.out(torch.cat([orig, x], dim=1)))   


class EDSRImageModifier(GeneratorModule):
    def __init__(self, nf=32):
        super().__init__() 
        rn, lr_cut = get_pretrained_resnet_base()

        self.rn = rn
        set_trainable(rn, False)
        self.lr_cut = lr_cut
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        
        self.up1 = UpSampleBlock(256, nf, 16)  #256 in
        self.up2 = UpSampleBlock(128, nf, 8)  #128 in
        self.up3 = UpSampleBlock(64, nf, 4)    #64 in
        self.up4 = UpSampleBlock(64, nf, 2)   #64 in  
        nf_up = nf*4+3
 
        mid_layers = []
        mid_layers += [ConvBlock(nf_up,nf, bn=True, actn=False)]
        
        for i in range(8): 
            mid_layers.append(
                ResSequential(
                [ConvBlock(nf, nf, actn=True, bn=False), 
                 ConvBlock(nf, nf, actn=False, bn=False)], 0.1)
            )
            
        mid_layers += [nn.BatchNorm2d(nf), ConvBlock(nf, 3, bn=False, actn=False)]
        self.upconv = nn.Sequential(*mid_layers)
             
        out_layers = []
        out_layers += [ConvBlock(6, 3, bn=False, actn=False)]
        self.out = nn.Sequential(*out_layers)

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]
        
    def forward(self, x): 
        self.rn(x)
        x1 = self.up1(self.sfs[3].features)
        x2 = self.up2(self.sfs[2].features)
        x3 = self.up3(self.sfs[1].features)
        x4 = self.up4(self.sfs[0].features) 
        x5 = self.upconv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return F.tanh(self.out(torch.cat([x, x5], dim=1)))    


class MinimalEDSRImageModifier(GeneratorModule):
    def __init__(self, nf=128):
        super().__init__() 

        layers = []
        layers += [ConvBlock(3,nf, bn=True, actn=False)]
        
        for i in range(10): 
            layers.append(
                ResSequential(
                [ConvBlock(nf, nf, actn=True, bn=False), 
                 ConvBlock(nf, nf, actn=False, bn=False)], 0.1)
            )
            
        layers += [nn.BatchNorm2d(nf), ConvBlock(nf, 3, bn=False, actn=False)]
        layers += [ConvBlock(3, 3, bn=False, actn=False)]
        self.out = nn.Sequential(*layers)

    def get_layer_groups(self)->[]:
        return children(self)
        
    def forward(self, x): 
        return F.tanh(self.out(x))   

class Unet34(GeneratorModule):  
    def __init__(self, nf=256):
        super().__init__()
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
        self.up1 = UnetBlock(512,256,nf)
        self.up2 = UnetBlock(nf,128,nf)
        self.up3 = UnetBlock(nf,64,nf)
        self.up4 = UnetBlock(nf,64,nf)
        self.up5 = UpSampleBlock(nf, nf, 2)      
        self.out= nn.Sequential(nn.BatchNorm2d(nf), ConvBlock(nf, 3, ks=1, actn=False, bn=False))
           
    def forward(self, x):
        x = F.relu(self.rn(x))
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)
        x = self.out(x)
        return x
    
    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]
    
    def close(self):
        for sf in self.sfs: 
            sf.remove()

class GeneratorModelWrapper():
    def __init__(self, model: GeneratorModule, name:str):
        self.model = to_gpu(model)
        self.name = name

    def get_layer_groups(self, precompute):
        return self.model.get_layer_groups()
from fasterai.modules import *
from fasterai.generators import *
from torch import autograd
from collections import Iterable
import torch.utils.hooks as hooks


class CriticModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def freeze_to(self, n):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)
    
    def set_trainable(self, trainable: bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self)->[]:
        pass

class AltFeatureCritic(CriticModule):
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  

        self.feature_eval_2 = DCCritic(128*2, 32, 1, sz//8, True)
        self.feature_eval_3 = DCCritic(64*2, 32, 2, sz//4, True)
        self.feature_eval_4 = DCCritic(64*2, 32, 4, sz//2, True) 
        self.pixel_eval = DCCritic(6, 32, 8, sz, True)

        nf = 32+64+128+256+512
        self.pre_final_eval = DCCritic(nf, nf//4, 4, sz//16, True)                                                   
        self.final_eval = nn.Sequential(
            Flatten(),
            nn.Linear(nf, 1))

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = torch.cat([x1, y1], dim=1)
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4= self.feature_eval_4(torch.cat([x4, y4], dim=1))
        
        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, p], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class AltFeatureCritic2(CriticModule):           
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  


        self.feature_eval_1 = DCCritic(256*2, 64, 1, sz//16, True)
        self.feature_eval_2 = DCCritic(128*2, 32, 2, sz//8, True)
        self.feature_eval_3 = DCCritic(64*2, 32, 4, sz//4, True)
        self.feature_eval_4 = DCCritic(64*2, 32, 8, sz//2, True) 
        self.pixel_eval = DCCritic(6, 64, 16, sz, True)

        nf = 64*16+32*8+32*4+32*2+64

        self.pre_final_eval = nn.Sequential(
            nn.Conv2d(nf, nf*2, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([nf*2, sz//64, sz//64,]))

        self.final_eval = nn.Conv2d(nf*2, 1, kernel_size=sz//64, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))
  
        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, p], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class AltFeatureCritic12(CriticModule): 
    def _generate_eval_layers(self, nf_in, nf_mid, sz):
        layers = [] 
        min_size = math.log(self.sz,2)
        csize,cndf = sz,nf_mid

        if csize > min_size:
            layers.append(DownSampleResBlock(ni=nf_in, nf=cndf, bn=False, dropout=0.2))
            csize = int(csize//2)
        else:
            layers.append(ConvBlock(nf_in, nf_mid, bn=False))
            layers.append(nn.Dropout2d(0.2))
        
        layers.append(ResBlock(nf=cndf, ks=3, bn=False))

        while csize > min_size:
            layers.append(DownSampleResBlock(ni=cndf, nf=cndf*2, bn=False))
            cndf = int(cndf*2)
            csize = int(csize//2)
            
        return nn.Sequential(*layers)
            
    def __init__(self, sz, filter_scale:float = 1.0):
        super().__init__()
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6,7]]
        self.sz = sz
        self.feature_eval_0 = self._generate_eval_layers(512*2, int(512*filter_scale), sz//32)
        self.feature_eval_1 = self._generate_eval_layers(256*2, int(256*filter_scale), sz//16)
        self.feature_eval_2 = self._generate_eval_layers(128*2, int(128*filter_scale), sz//8)
        self.feature_eval_3 = self._generate_eval_layers(64*2,  int(64*filter_scale), sz//4)
        self.feature_eval_4 = self._generate_eval_layers(64*2, int(64*filter_scale), sz//2)     
        self.pixel_eval = self._generate_eval_layers(6, int(128*filter_scale), sz)

        nf_mid = int((512 + 256 + 128*2+ 64*4 + 64*8+ 128*16)*filter_scale)

        self.mid = ResBlock(nf=nf_mid, ks=sz//32, bn=False)
        self.out = nn.Conv2d(nf_mid, 1, 1, padding=0, bias=False)
        
    def forward(self, input, orig):
        self.rn(orig)
        x0 = self.sfs[4].features
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y0 = self.sfs[4].features
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f0 = self.feature_eval_0(torch.cat([x0, y0], dim=1))
        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))
        p = self.pixel_eval(torch.cat([orig, input], dim=1))

        before_last = self.mid(torch.cat([f0,f1,f2,f3,f4,p], dim=1))
        x = self.out(before_last)
        return x, before_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class AltFeatureCritic11(CriticModule): 

    def _generate_eval_layers(self, nf_in, nf_mid, sz):
        layers = [] 
        layers.append(ConvBlock(nf_in, nf_mid, 4, 2))
        layers.append(nn.Dropout(0.2))
        csize,cndf = sz//2,nf_mid
        layers.append(ConvBlock(cndf, cndf, 3, 1))
        layers.append(nn.Dropout(0.5))
        min_size = 8

        while csize > min_size:
            layers.append(ConvBlock(cndf, cndf*2, 4, 2))
            cndf = int(cndf*2)
            csize = int(csize//2)
            if csize <= min_size: layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Conv2d(cndf, 1, 4, padding=0, bias=False))    
        return nn.Sequential(*layers) 
            
    def __init__(self, sz):
        super().__init__()
         
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
        
        self.feature_eval_1 = self._generate_eval_layers(256*2, 256, sz//16)
        self.feature_eval_2 = self._generate_eval_layers(128*2, 128, sz//8)
        self.feature_eval_3 = self._generate_eval_layers(64*2, 64, sz//4)
        self.feature_eval_4 = self._generate_eval_layers(64*2, 64, sz//2)     
        self.pixel_eval = self._generate_eval_layers(6, 64, sz)

        self.out = nn.Conv2d(5, 1, 4, padding=0, bias=False)
        
    def forward(self, input, orig):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))

        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        before_last = torch.cat([f1,f2,f3,f4,p], dim=1)
        x = self.out(before_last)
        return x, before_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class AltFeatureCritic10(CriticModule): 

    def _generate_eval_layers(self, nf_in, nf_mid, sz):
        layers = [] 
        layers.append(ConvBlock(nf_in, nf_mid, 4, 2, bn=False))
        layers.append(nn.Dropout(0.2))
        csize,cndf = sz//2,nf_mid
        layers.append(ConvBlock(cndf, cndf, 3, 1, bn=False))
        layers.append(nn.Dropout(0.5))
        min_size = 8

        while csize > min_size:
            layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            cndf = int(cndf*2)
            csize = int(csize//2)
            layers.append(nn.Dropout(0.5))
        
        layers.append(nn.Conv2d(cndf, 1, 4, padding=0, bias=False))    
        return nn.Sequential(*layers) 
            
    def __init__(self, sz):
        super().__init__()
         
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
        
        self.feature_eval_1 = self._generate_eval_layers(256*2, 256, sz//16)
        self.feature_eval_2 = self._generate_eval_layers(128*2, 128, sz//8)
        self.feature_eval_3 = self._generate_eval_layers(64*2, 64, sz//4)
        self.feature_eval_4 = self._generate_eval_layers(64*2, 64, sz//2)     
        self.pixel_eval = self._generate_eval_layers(6, 64, sz)

        self.out = nn.Conv2d(5, 1, 4, padding=0, bias=False)
        
    def forward(self, input, orig):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))

        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        before_last = torch.cat([f1,f2,f3,f4,p], dim=1)
        x = self.out(before_last)
        return x, before_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class AltFeatureCritic9(CriticModule): 

    def _generate_eval_layers(self, nf_in, nf_mid, sz):
        layers = [] 
        layers.append(ConvBlock(nf_in, nf_mid, 4, 2, bn=False))
        layers.append(nn.Dropout(0.2))
        csize,cndf = sz//2,nf_mid
        layers.append(nn.LayerNorm([cndf, csize, csize]))
        layers.append(ConvBlock(cndf, cndf, 3, 1))
        layers.append(nn.Dropout(0.5))
        layers.append(nn.LayerNorm([cndf, csize, csize]))
        min_size = 8

        while csize > min_size:
            layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            layers.append(nn.Dropout(0.5))
            cndf = int(cndf*2)
            csize = int(csize//2)
            layers.append(nn.LayerNorm([cndf, csize, csize]))
        
        layers.append(nn.Conv2d(cndf, 1, 4, padding=0, bias=False))    
        return nn.Sequential(*layers) 
            
    def __init__(self, sz):
        super().__init__()
         
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
        
        self.feature_eval_1 = self._generate_eval_layers(256*2, 256, sz//16)
        self.feature_eval_2 = self._generate_eval_layers(128*2, 128, sz//8)
        self.feature_eval_3 = self._generate_eval_layers(64*2, 64, sz//4)
        self.feature_eval_4 = self._generate_eval_layers(64*2, 64, sz//2)     
        self.pixel_eval = self._generate_eval_layers(6, 64, sz)

        self.out = nn.Conv2d(5, 1, 4, padding=0, bias=False)
        
    def forward(self, input, orig):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))

        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        before_last = torch.cat([f1,f2,f3,f4,p], dim=1)
        x = self.out(before_last)
        return x, before_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class AltFeatureCritic8(CriticModule): 
    @staticmethod
    def generate_eval_layers(nf_in, nf_mid, sz):
        layers = [] 
        layers.append(ConvBlock(nf_in, nf_mid, 4, 2, bn=False))
        csize,cndf = sz//2,nf_mid
        #layers.append(nn.LayerNorm([cndf, csize, csize]))
        layers.append(ConvBlock(cndf, cndf, 3, 1))
        #layers.append(nn.LayerNorm([cndf, csize, csize]))

        while csize > 8:
            layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            cndf = int(cndf*2)
            csize = int(csize//2)
            #layers.append(nn.LayerNorm([cndf, csize, csize]))
        
        layers.append(nn.Conv2d(cndf, 1, 4, padding=0, bias=False))    
        return nn.Sequential(*layers) 
            
    def __init__(self, sz):
        super().__init__()
         
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
        
        self.feature_eval_1 = AltFeatureCritic8.generate_eval_layers(256*2, 256, sz//16)
        self.feature_eval_2 = AltFeatureCritic8.generate_eval_layers(128*2, 128, sz//8)
        self.feature_eval_3 = AltFeatureCritic8.generate_eval_layers(64*2, 64, sz//4)
        self.feature_eval_4 = AltFeatureCritic8.generate_eval_layers(64*2, 64, sz//2)     
        self.pixel_eval = AltFeatureCritic8.generate_eval_layers(6, 64, sz)

        self.out = nn.Conv2d(5, 1, 4, padding=0, bias=False)
        
    def forward(self, input, orig):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))

        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        before_last = torch.cat([f1,f2,f3,f4,p], dim=1)
        x = self.out(before_last)
        return x, before_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class AltFeatureCritic7(CriticModule): 

    def __init__(self, size, nf_up=32, nf_mid=64):
        super().__init__()
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]
  
        self.up1 = UpSampleBlock(256,nf_up, 16) #256 in
        self.up2 = UpSampleBlock(128,nf_up, 8)  #128 in
        self.up3 = UpSampleBlock(64,nf_up, 4)    #64 in
        self.up4 = UpSampleBlock(64, nf_up, 2)   #64 in  
        self.raw_up = ConvBlock(3, nf_up, 3, 1, bn=False)
        
        mid_layers = [ConvBlock(nf_up*10, nf_mid, 3, 1, bn=False)]
        mid_layers.extend([ConvBlock(nf_mid, nf_mid, 3, 1, bn=False) for t in range(3)])     
        self.mid = nn.Sequential(*mid_layers)  
        
        pyr_layers = []
        csize = size
        cndf = nf_mid
        
        while csize > 8:
            pyr_layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            cndf *= 2; csize /= 2
        self.pyramid = nn.Sequential(*pyr_layers)
        
        self.final = nn.Conv2d(cndf, 1, csize, padding=0, bias=False)
        
    def forward(self, input, orig):
        self.rn(orig)
        x1 = self.up1(self.sfs[3].features)
        x2 = self.up2(self.sfs[2].features)
        x3 = self.up3(self.sfs[1].features)
        x4 = self.up4(self.sfs[0].features) 
        orig_features =  torch.cat([x1, x2, x3, x4], dim=1)
        
        self.rn(input)
        x1 = self.up1(self.sfs[3].features)
        x2 = self.up2(self.sfs[2].features)
        x3 = self.up3(self.sfs[1].features)
        x4 = self.up4(self.sfs[0].features) 
        result_features = torch.cat([x1, x2, x3, x4], dim=1)
        
        orig_up = self.raw_up(orig)
        input_up = self.raw_up(input)
        
        combined_features = torch.cat([orig_up, input_up, orig_features, result_features], dim=1)      
        x = self.mid(combined_features)
        before_last = self.pyramid(x)
        return self.final(before_last), before_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

 
class AltFeatureCritic6(CriticModule):     
    def _down_filter_block(self, ni:int, nf:int, ks:int):
        stride = 1
        pad = ks//2//stride
        layers=[nn.Conv2d(ni, nf, ks, stride, pad)]
        layers+=[nn.LeakyReLU()]
        return nn.Sequential(*layers)


    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  

        self.down_1 = nn.Sequential(
            self._down_filter_block(256*2, 256, 3),
            self._down_filter_block(256, 128, 3),
            self._down_filter_block(128, 64, 3),
            self._down_filter_block(64, 32, 3))


        self.down_2 = nn.Sequential(
            self._down_filter_block(256, 128, 3),
            self._down_filter_block(128, 64, 3),
            self._down_filter_block(64, 32, 3))

        self.down_3 = nn.Sequential(
            self._down_filter_block(128, 64, 3),
            self._down_filter_block(64, 32, 3))

        self.down_4 = nn.Sequential(
            self._down_filter_block(128, 64, 3),
            self._down_filter_block(64, 32, 3))

        self.feature_eval_1 = DCCritic(32, 32, 1, sz//16, True, False)
        self.feature_eval_2 = DCCritic(32, 32, 2, sz//8, True, False)
        self.feature_eval_3 = DCCritic(32, 32, 4, sz//4, True, False)
        self.feature_eval_4 = DCCritic(32, 32, 8, sz//2, True, False) 
        self.pixel_eval = DCCritic(6, 64, 16, sz, True, False)

        nf = 32*15 + 64*16

        self.pre_final_eval = DCCritic(nf, 2048, 2, sz//32, True, False)
        self.final_eval = nn.Conv2d(4096, 1, kernel_size=sz//128, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        d1 = self.down_1(torch.cat([x1, y1], dim=1))
        d2 = self.down_2(torch.cat([x2, y2], dim=1))
        d3 = self.down_3(torch.cat([x3, y3], dim=1)) 
        d4 = self.down_4(torch.cat([x4, y4], dim=1))

        f1 = self.feature_eval_1(d1)
        f2 = self.feature_eval_2(d2)
        f3 = self.feature_eval_3(d3)
        f4 = self.feature_eval_4(d4)
        f5 = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, f5], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class AltFeatureCritic5(CriticModule):           
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  


        self.feature_eval_1 = DCCritic(256*2, 16, 1, sz//16, True, False)
        self.feature_eval_2 = DCCritic(128*2, 16, 2, sz//8, True, False)
        self.feature_eval_3 = DCCritic(64*2, 16, 4, sz//4, True, False)
        self.feature_eval_4 = DCCritic(64*2, 16, 8, sz//2, True, False) 
        self.pixel_eval = DCCritic(6, 128, 16, sz, True, False)

        nf = 16*1+16*2+16*4+16*8+128*16

        self.pre_final_eval = DCCritic(nf, 2048, 2, sz//32, True, False)

        self.final_eval = nn.Conv2d(4096, 1, kernel_size=sz//128, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))
  
        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, p], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class AltFeatureCritic4(CriticModule):           
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  


        self.feature_eval_1 = DCCritic(256*2, 16, 1, sz//16, True, False)
        self.feature_eval_2 = DCCritic(128*2, 16, 2, sz//8, True, False)
        self.feature_eval_3 = DCCritic(64*2, 16, 4, sz//4, True, False)
        self.feature_eval_4 = DCCritic(64*2, 16, 8, sz//2, True, False) 
        self.pixel_eval = DCCritic(6, 64, 16, sz, True, False)

        nf = 16*1+16*2+16*4+16*8+64*16

        self.pre_final_eval = DCCritic(nf, 2048, 2, sz//32, True, False)

        self.final_eval = nn.Conv2d(4096, 1, kernel_size=sz//128, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))
  
        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, p], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class AltFeatureCritic3(CriticModule):           
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  


        self.feature_eval_1 = DCCritic(256*2, 64, 1, sz//16, True, False)
        self.feature_eval_2 = DCCritic(128*2, 32, 2, sz//8, True, False)
        self.feature_eval_3 = DCCritic(64*2, 32, 4, sz//4, True, False)
        self.feature_eval_4 = DCCritic(64*2, 32, 8, sz//2, True, False) 
        self.pixel_eval = DCCritic(6, 64, 16, sz, True, False)

        nf = 64*16+32*8+32*4+32*2+64

        self.pre_final_eval = nn.Sequential(
            nn.Conv2d(nf, nf*2, 4, 2, padding=1),
            nn.LeakyReLU())

        self.final_eval = nn.Conv2d(nf*2, 1, kernel_size=sz//64, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))
  
        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, p], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class SimpleResnetCritic(CriticModule):  
    def _initial_block(self, ni:int, nf:int): 
        stride = 1
        ks=3
        pad = ks//2//stride
        layers=[nn.Conv2d(ni, nf, ks, padding=pad, stride=stride)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.BatchNorm2d(nf)]
        layers+=[nn.Dropout2d(0.2)]
        layers+=[ConvPoolMean(nf, nf)] 
        return nn.Sequential(*layers)

    def _residual_block(self, ni:int, nf:int, ks:int):
        stride = 1
        pad = ks//2//stride
        layers = [nn.BatchNorm2d(ni)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.Conv2d(ni, nf, ks, stride, pad)]
        layers+= [nn.BatchNorm2d(nf)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.Dropout2d(0.5)]
        layers+=[nn.Conv2d(nf, nf, ks, stride, pad)]
        return ResSequential(layers)

    def __init__(self, sz:int, nf:int=1024):
        super().__init__()      
        self.rn, self.lr_cut = get_pretrained_resnet_base(1)
        set_trainable(self.rn, False)

        ni = 512
        self.init = self._initial_block(ni, nf)
        self.init_shortcut = MeanPoolConv(ni, nf)

        mid_layers = [ResBlock(nf)]
        mid_layers += [ResBlock(nf)]
        mid_layers += [ResBlock(nf)]

        self.mid = nn.Sequential(*mid_layers)
        self.final_eval = nn.Conv2d(nf, 1, kernel_size=sz//32, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        x1 = self.rn(input)
        x2 = self.rn(orig)
        x = torch.cat([x1, x2], dim=1)
        x = self.init(x) + self.init_shortcut(x)
        pre_last = self.mid(x)
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]


class ResnetFeatureCritic(CriticModule):  

    def _initial_block(self, ni:int, nf:int): 
        stride = 1
        ks=3
        pad = ks//2//stride
        layers=[nn.Conv2d(ni, nf, ks, padding=pad, stride=stride)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.BatchNorm2d(nf)]
        layers+=[nn.Dropout2d(0.2)]
        layers+=[ConvPoolMean(nf, nf)] 
        return nn.Sequential(*layers)

    def _residual_block(self, ni:int, nf:int, ks:int, sz:int):
        stride = 1
        pad = ks//2//stride
        layers = [nn.BatchNorm2d(ni)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.Conv2d(ni, nf, ks, stride, pad)]
        layers+= [nn.BatchNorm2d(nf)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.Dropout2d(0.5)]
        layers+=[nn.Conv2d(nf, nf, ks, stride, pad)]
        return ResSequential(layers)

    def _down_sample_block(self, ni:int, nf:int, ks:int, sz:int):
        stride = 1
        pad = ks//2//stride
        layers = [nn.BatchNorm2d(nf)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.Conv2d(ni, ni, ks, padding=pad, stride=stride)]
        layers+=[nn.BatchNorm2d(nf)]
        layers+=[nn.LeakyReLU()]
        layers+=[nn.Dropout2d(0.5)]
        layers+=[ConvPoolMean(ni, nf)] 
        return nn.Sequential(*layers)
        

    def __init__(self, sz:int, nf:int=128):
        super().__init__() 
        self.dropoutp = 0.5       
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  


        self.up1 = UpSampleBlock(256, 32, 16)  #256 in
        self.up2 = UpSampleBlock(128, 32, 8)  #128 in
        self.up3 = UpSampleBlock(64, 32, 4)    #64 in
        self.up4 = UpSampleBlock(64, 32, 2)   #64 in  
        nf_up = 32*8+6
 
        self.init = self._initial_block(nf_up, nf)
        self.init_shortcut = MeanPoolConv(nf_up, nf)

        self.down = self._down_sample_block(nf, nf, 3, sz//2)
        self.down_shortcut = MeanPoolConv(nf, nf)

        mid_layers = [self._residual_block(nf, nf, 3, sz//2)]
        mid_layers += [self._residual_block(nf, nf, 3, sz//2)]
        mid_layers += [self._residual_block(nf, nf, 3, sz//2)]

        self.mid = nn.Sequential(*mid_layers)
        self.final_eval = nn.Conv2d(nf, 1, kernel_size=sz//4, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.up1(self.sfs[3].features)
        x2 = self.up2(self.sfs[2].features)
        x3 = self.up3(self.sfs[1].features)
        x4 = self.up4(self.sfs[0].features)
        
        self.rn(input)
        y1 = self.up1(self.sfs[3].features)
        y2 = self.up2(self.sfs[2].features)
        y3 = self.up3(self.sfs[1].features)
        y4 = self.up4(self.sfs[0].features)

        x = torch.cat([input, orig, x1, x2, x3, x4, y1, y2, y3, y4], dim=1)

        x = self.init(x) + self.init_shortcut(x)
        x = self.down(x) + self.down_shortcut(x)

        pre_last = self.mid(x)
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class DCCritic(CriticModule):
    def __init__(self, ni:int, nf:int, scale:int, sz:int, skip_last_layer=False, normalize=True):
        super().__init__()

        assert (math.log(scale,2)).is_integer()
        self.initial = ConvBlock(ni, nf, 4, 2, bn=False)
        cndf = nf
        csize = sz//2

        mid_layers =  []
        if normalize: mid_layers.append(nn.LayerNorm([cndf, csize, csize]))   
        mid_layers.append(ConvBlock(cndf, cndf, 3, 1, bn=False))
        if normalize: mid_layers.append(nn.LayerNorm([cndf, csize, csize]))

        self.mid=nn.Sequential(*mid_layers)

        out_layers=[]
        scale_count = 0

        for i in range(int(math.log(scale,2))):
            out_layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            cndf = int(cndf*2)
            csize = int(csize//2)
            if normalize: out_layers.append(nn.LayerNorm([cndf, csize, csize]))
        
        if not skip_last_layer:
            out_layers.append(nn.Conv2d(cndf, 1, kernel_size=csize, stride=1, padding=0, bias=False))   

        self.out = nn.Sequential(*out_layers) 

    def get_layer_groups(self)->[]:
        return children(self)
    
    def forward(self, x):
        x=self.initial(x)
        x=self.mid(x)
        return self.out(x)

#Works with sz >= 128
class FeatureCritic(CriticModule):           
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]  


        self.feature_eval_1 = DCCritic(256*2, 256, 1, sz//16, True)
        self.feature_eval_2 = DCCritic(128*2, 128, 2, sz//8, True)
        self.feature_eval_3 = DCCritic(64*2, 64, 4, sz//4, True)
        self.feature_eval_4 = DCCritic(64*2, 64, 8, sz//2, True) 
        self.pixel_eval = DCCritic(6, 32, 16, sz, True)

        nf = 256 + 128*2 + 64*4 + 64*8 + 32*16

        self.pre_final_eval = nn.Sequential(
            nn.Conv2d(nf, nf*2, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.LayerNorm([nf*2, sz//64, sz//64,]))

        self.final_eval = nn.Conv2d(nf*2, 1, kernel_size=sz//64, stride=1, padding=0, bias=False)  

        
    def forward(self, input: torch.Tensor, orig: torch.Tensor):
        self.rn(orig)
        x1 = self.sfs[3].features
        x2 = self.sfs[2].features
        x3 = self.sfs[1].features
        x4 = self.sfs[0].features
        
        self.rn(input)
        y1 = self.sfs[3].features
        y2 = self.sfs[2].features
        y3 = self.sfs[1].features
        y4 = self.sfs[0].features 

        f1 = self.feature_eval_1(torch.cat([x1, y1], dim=1))
        f2 = self.feature_eval_2(torch.cat([x2, y2], dim=1))
        f3 = self.feature_eval_3(torch.cat([x3, y3], dim=1))
        f4 = self.feature_eval_4(torch.cat([x4, y4], dim=1))
  
        p = self.pixel_eval(torch.cat([orig, input], dim=1))
        
        pre_last = self.pre_final_eval(torch.cat([f1, f2, f3, f4, p], dim=1))
        final = self.final_eval(pre_last)
        return final, pre_last

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class WGANGenTrainingResult():
    def __init__(self, gcost: np.array, gcount: int):
        self.gcost=gcost
        self.gcount=gcount

class WGANCriticTrainingResult():
    def __init__(self, wdist: np.array, gpenalty: np.array, dreal: np.array, dfake: np.array, dcost: np.array, conpenalty: np.array):
        self.wdist=wdist
        self.gpenalty=gpenalty
        self.dreal=dreal
        self.dfake=dfake
        self.dcost=dcost
        self.conpenalty=conpenalty

class WGANTrainer():
    def __init__(self, netD: nn.Module, netG: GeneratorModule, md: ImageData, 
            bs:int, sz:int, dpath: Path, gpath: Path, gplambda=10, citers=5):
        self.netD = netD
        self.netG = netG
        self.md = md
        self.bs = bs
        self.sz = sz
        self.dpath = dpath
        self.gpath = gpath
        self.gplambda = gplambda
        self.citers = citers
        self._train_loop_hooks = OrderedDict()

    def register_train_loop_hook(self, hook):
        handle = hooks.RemovableHandle(self._train_loop_hooks)
        self._train_loop_hooks[handle.id] = hook
        return handle

    def train(self, lrs_critic:[int], lrs_gen:[int], clr_critic: (int)=(20,10), clr_gen: (int)=(20,10), 
            cycle_len:int =1, epochs: int=1):

        self.gen_sched = self._generate_clr_sched(self.netG, clr_gen, lrs_gen, cycle_len)
        self.critic_sched = self._generate_clr_sched(self.netD, clr_critic, lrs_critic, cycle_len)

        gcount = 0
        self.critic_sched.on_train_begin()
        self.gen_sched.on_train_begin()

        for epoch in trange(epochs):
            gcount = self._train_one_epoch(gcount)

    def _get_raw_noise_loss(self, orig_image: torch.Tensor):
        noise_image = self._create_noise_batch()
        return self.netD(noise_image, orig_image)

    def _create_noise_batch(self): 
        raw_random = V(torch.randn(self.bs, 3, self.sz,self.sz).normal_(0, 1))
        return F.tanh(raw_random)

    def _generate_clr_sched(self, model, use_clr_beta: (int), lrs: [int], cycle_len: int):
        wds = 1e-7
        opt_fn = partial(optim.Adam, betas=(0., 0.9))
        layer_opt = LayerOptimizer(opt_fn, model.get_layer_groups(), lrs, wds)
        div,pct = use_clr_beta[:2]
        moms = use_clr_beta[2:] if len(use_clr_beta) > 3 else None
        cycle_end =  None
        return CircularLR_beta(layer_opt, len(self.md.trn_dl)*cycle_len, on_cycle_end=cycle_end, div=div, pct=pct, momentums=moms)
    
    def _train_one_epoch(self, gcount: int)->int:
        self.netD.train()
        self.netG.train()
        data_iter = iter(self.md.trn_dl)
        n = len(self.md.trn_dl)
        with tqdm(total=n) as pbar:
            while True:
                cresult = self._train_critic(gcount, data_iter, pbar)
                
                if cresult is None:
                    break
                
                gresult = self._train_generator(gcount, data_iter, pbar, cresult)
                gcount = gresult.gcount

                if gresult is None:
                    break

                self._save_if_applicable(gresult, cresult)
                self._call_train_loop_hooks(gresult, cresult)
        
        return gcount

    def _call_train_loop_hooks(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        for hook in self._train_loop_hooks.values():
            hook_result = hook(self, gresult, cresult)
            if hook_result is not None:
                raise RuntimeError(
                    "train loop hooks hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))

    def _get_next_training_images(self, data_iter: Iterable)->(torch.Tensor,torch.Tensor):
        x, y = next(data_iter, (None, None))
        if x is None:
            return (None, None)
        orig_image = V(x)
        real_image = V(y) 
        return (orig_image, real_image)


    def _calculate_wdist(self, orig_image: torch.Tensor, real_image: torch.Tensor, fake_image: torch.Tensor)->torch.Tensor:
        dreal = self._get_dscore(real_image, orig_image)
        dfake = self._get_dscore(V(fake_image.data), orig_image)
        wdist = dfake - dreal
        return wdist, dfake, dreal

    def _is_equilibrium(self, cresult: WGANCriticTrainingResult):
        dreal = cresult.dreal
        dfake = cresult.dfake

        if dreal < dfake:
            return False
            
        return abs(dreal + dfake) < (abs(dreal) + abs(dfake))*0.30

    def _train_critic(self, gcount: int, data_iter: Iterable, pbar: tqdm)->WGANCriticTrainingResult:
        self.netD.set_trainable(True)
        self.netG.set_trainable(False)
        j = 0
        cresult=None

        equilib = False

        while j < self.citers and j < 500:
            orig_image, real_image = self._get_next_training_images(data_iter)
            if orig_image is None:
                return cresult
            j += 1
            #To help boost critic early on, we're doing both noise and generator based training, since
            #the generator never actually starts by creating noise
            #self._train_critic_once(orig_image, real_image, noise_gen=True)
            cresult = self._train_critic_once(orig_image, real_image, noise_gen=False)
            pbar.update()
            equilib = self._is_equilibrium(cresult)

        
        return cresult

    def _train_critic_once(self, orig_image: torch.Tensor, real_image: torch.Tensor, noise_gen:bool= False)->WGANCriticTrainingResult:                     
        #Higher == Real
        fake_image = self._create_noise_batch() if noise_gen else self.netG(orig_image)
        wdist, dfake, dreal = self._calculate_wdist(orig_image, real_image, fake_image)
        self.netD.zero_grad()        
        gpenalty = self._calc_gradient_penalty(real_image.data, fake_image.data, orig_image)     
        conpenalty = self._consistency_penalty(real_image.data, orig_image)          
        dcost = dfake - dreal + gpenalty + conpenalty
        dcost.backward()
        self.critic_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(dcost))
        self.gen_sched.on_batch_end(to_np(-dfake))
        return WGANCriticTrainingResult(to_np(wdist), to_np(gpenalty), to_np(dreal), to_np(dfake), to_np(dcost), to_np(conpenalty))
    
    def _train_generator(self, gcount: int, data_iter: Iterable, pbar: tqdm, cresult: WGANCriticTrainingResult)->WGANGenTrainingResult:
        orig_image, real_image = self._get_next_training_images(data_iter)   
        if orig_image is None:
            return None
        gcount += 1   
        gresult = self._train_generator_once(orig_image, real_image, gcount, cresult)       
        pbar.update() 
        return gresult

    def _train_generator_once(self, orig_image: torch.Tensor, real_image: torch.Tensor, 
            gcount: int, cresult: WGANCriticTrainingResult)->WGANGenTrainingResult:
        self.netD.set_trainable(False)
        self.netG.set_trainable(True)
        self.netG.zero_grad()         
        fake_image = self.netG(orig_image)
        gcost = -self._get_dscore(fake_image, orig_image)
        gcost.backward()
        self.gen_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(cresult.dcost))
        self.gen_sched.on_batch_end(to_np(gcost))
        return WGANGenTrainingResult(to_np(gcost), gcount)

    def _save_if_applicable(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        if cresult is None or gresult is None:
            return

        if gresult.gcount % 10 == 0:
            save_model(self.netD, self.dpath)
            save_model(self.netG, self.gpath)

    def _get_dscore(self, new_image: torch.Tensor, orig_image: torch.Tensor):
        #return self._normalize_loss(orig_image, self.netD(new_image, orig_image))
        final, _ = self.netD(new_image, orig_image)
        return final.mean()

    def _calc_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, orig_data: torch.Tensor)->torch.Tensor:
        alpha = torch.rand(self.bs, 1)
        alpha = alpha.expand(self.bs, real_data.nelement()//self.bs).contiguous().view(self.bs, 3, self.sz, self.sz)
        alpha = alpha.cuda()
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self._get_dscore(interpolates, orig_data)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gplambda
        return gradient_penalty

    def _consistency_penalty(self, real_data: torch.Tensor, orig_data: torch.Tensor):
        d1, d_1 = self.netD(real_data, orig_data)
        d2, d_2 = self.netD(real_data, orig_data)
        consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * (d_1 - d_2).norm(2, dim=1)
        return consistency_term.mean()
from fasterai.modules import *
from fasterai.visualize import *
from torch import autograd
from collections import Iterable
from abc import ABC, abstractmethod

class DCCritic(nn.Module):
    def __init__(self, ni:int, nf:int, sz:int):
        super().__init__()

        self.initial = ConvBlock(ni, nf, 4, 2, bn=False)
        csize,cndf = sz//2,nf

        self.mid=nn.Sequential(
            nn.LayerNorm([cndf, csize, csize]), 
            ConvBlock(cndf, cndf, 3, 1, bn=False),
            nn.LayerNorm([cndf, csize, csize]))

        out_layers=[]
        while csize > 4:
            out_layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            cndf = int(cndf*2)
            csize = int(csize//2)
            out_layers.append(nn.LayerNorm([cndf, csize, csize]))
        
        out_layers.append(nn.Conv2d(cndf, 1, csize, padding=0, bias=False))        
        self.out = nn.Sequential(*out_layers) 
    
    def forward(self, x):
        x=self.initial(x)
        x=self.mid(x)
        return self.out(x)


class FeatureCritic(nn.Module):
    def set_trainable(self, trainable):
        set_trainable(self, trainable)
        set_trainable(self.rn, False)
                
    def __init__(self, sz:int, nf:int=128):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]      
        self.feature_eval_1 = DCCritic(256*2, nf, sz//16)
        self.feature_eval_2 = DCCritic(128*2, nf, sz//8)
        self.feature_eval_3 = DCCritic(64*2, nf, sz//4)
        self.feature_eval_4 = DCCritic(64*2, nf, sz//2)     
        self.pixel_eval = DCCritic(6, nf, sz)

        
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
        return f1.mean() + f2.mean() + f3.mean()  + f4.mean() + p.mean()

class WGANGenTrainingResult():
    def __init__(self, gcost: np.array, gcount: int):
        self.gcost=gcost
        self.gcount=gcount

class WGANCriticTrainingResult():
    def __init__(self, wdist: np.array, gpenalty: np.array, dreal: np.array, dfake: np.array):
        self.wdist=wdist
        self.gpenalty=gpenalty
        self.dreal=dreal
        self.dfake=dfake

class GeneratorModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def set_trainable(self, trainable: bool):
        pass

class WGANTrainer():
    def __init__(self, netD: nn.Module, netG: GeneratorModule, md: ImageData, bs:int, sz:int, dpath: Path, gpath: Path, lr:float=1e-4):
        self.netD = netD
        self.netG = netG
        self.md = md
        self.bs = bs
        self.sz = sz
        self.lr = lr
        self.dpath = dpath
        self.gpath = gpath
        self.optimizerD = optim.Adam(filter(lambda p: p.requires_grad,netD.parameters()), lr=lr, betas=(0., 0.9))
        self.optimizerG = optim.Adam(filter(lambda p: p.requires_grad,netG.parameters()), lr=lr, betas=(0., 0.9))

    def train(self, niter: int=1, first_epoch=True):
        gcount = 0
        for epoch in trange(niter):
            gcount = self._train_one_epoch(gcount, first_epoch)
    
    def _train_one_epoch(self, gcount: int, first_epoch: bool)->int:
        self.netD.train()
        self.netG.train()
        data_iter = iter(self.md.trn_dl)
        n = len(self.md.trn_dl)
        with tqdm(total=n) as pbar:
            while True:
                cresult = self._train_critic(first_epoch, gcount, data_iter, pbar)
                
                if cresult is None:
                    break

                gresult = self._train_generator(gcount, data_iter, pbar)
                gcount = gresult.gcount

                if gresult is None:
                    break

                self._progress_update(gresult, cresult)
        
        return gcount


    def _get_num_critic_iters(self, first_epoch: bool, gcount: int)->int:
        return 100 if (first_epoch and (gcount < 25) or (gcount % 500 == 0)) else 5

    def _get_next_training_images(self, data_iter: Iterable)->(torch.Tensor,torch.Tensor):
        x, y = next(data_iter, (None, None))
        if x is None:
            return (None, None)
        orig_image = V(x)
        real_image = V(y) 
        return (orig_image, real_image)

    def _calculate_wdist(self, orig_image: torch.Tensor, real_image: torch.Tensor, fake_image: torch.Tensor)->torch.Tensor:
        dreal = self.netD(real_image, orig_image)
        dfake = self.netD(V(fake_image.data), orig_image)
        wdist = dfake - dreal
        return wdist, dfake, dreal

    def _train_critic(self, first_epoch: bool, gcount: int, data_iter: Iterable, pbar: tqdm)->WGANCriticTrainingResult:
        self.netD.set_trainable(True)
        self.netG.set_trainable(False)
        j = 0
        d_iters = self._get_num_critic_iters(first_epoch, gcount)
        cresult=None

        while (j<d_iters):
            orig_image, real_image = self._get_next_training_images(data_iter)
            if orig_image is None:
                return cresult
            j += 1
            cresult = self._train_critic_once(orig_image, real_image)
            pbar.update()
        
        return cresult

    def _train_critic_once(self, orig_image: torch.Tensor, real_image: torch.Tensor)->WGANCriticTrainingResult:                     
        #Higher == Real
        fake_image = self.netG(orig_image)
        wdist, dfake, dreal = self._calculate_wdist(orig_image, real_image, fake_image)
        self.netD.zero_grad()        
        gpenalty = self._calc_gradient_penalty(real_image.data, fake_image.data, orig_image)              
        disc_cost = dfake - dreal + gpenalty
        disc_cost.backward()
        self.optimizerD.step()
        return WGANCriticTrainingResult(to_np(wdist), to_np(gpenalty), to_np(dreal), to_np(dfake))
    
    def _train_generator(self, gcount: int, data_iter: Iterable, pbar: tqdm)->WGANGenTrainingResult:
        orig_image, real_image = self._get_next_training_images(data_iter)   
        if orig_image is None:
            return None
        gcount += 1   
        gresult = self._train_generator_once(orig_image, real_image, gcount)       
        pbar.update() 
        return gresult

    def _train_generator_once(self, orig_image: torch.Tensor, real_image: torch.Tensor, gcount: int)->WGANGenTrainingResult:
        self.netD.set_trainable(False)
        self.netG.set_trainable(True)
        self.netG.zero_grad()         
        fake_image = self.netG(orig_image)
        gcost  = -self.netD(fake_image, orig_image)
        gcost.backward()
        self.optimizerG.step()
        return WGANGenTrainingResult(to_np(gcost), gcount)

    def _progress_update(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        if cresult is None or gresult is None:
            return

        if gresult.gcount % 10 == 0:
            print(f'\nWDist {cresult.wdist}; RScore {cresult.dreal}; FScore {cresult.dfake}' + 
                f'; GCount: {gresult.gcount}; GPenalty: {cresult.gpenalty}; GCost: {gresult.gcost}')

        if gresult.gcount % 100 == 0:
            visualize_image_gen_model(self.md, self.netG, 500, 8)
            save_model(self.netD, self.dpath)
            save_model(self.netG, self.gpath)

    def _calc_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, orig_data: torch.Tensor)->torch.Tensor:
        lamda = 10 # Gradient penalty lambda hyperparameter
        alpha = torch.rand(self.bs, 1)
        alpha = alpha.expand(self.bs, real_data.nelement()//self.bs).contiguous().view(self.bs, 3, self.sz, self.sz)
        alpha = alpha.cuda()
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self.netD(interpolates, orig_data)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
        return gradient_penalty

class ResnetImageModifier(GeneratorModule): 
    def set_trainable(self, trainable: bool):
        set_trainable(self, trainable)
        set_trainable(self.rn, False)
        
    def __init__(self, nf:int=128):
        super().__init__() 
        
        self.rn, _ = get_pretrained_resnet_base(1)
        set_trainable(self.rn, False)
        
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
        
    def forward(self, orig): 
        x = self.rn(orig)
        x = self.color(x)
        return F.tanh(self.out(torch.cat([orig, x], dim=1)))   


class EDSRImageModifier(GeneratorModule):
    def set_trainable(self, trainable: bool):
        set_trainable(self, trainable)
        set_trainable(self.rn, False)
    
    def __init__(self):
        super().__init__() 
        rn, lr_cut = get_pretrained_resnet_base()

        self.rn = rn
        set_trainable(rn, False)
        self.lr_cut = lr_cut
        self.sfs = [SaveFeatures(rn[i]) for i in [2,4,5,6]]
        
        self.up1 = UpSampleBlock(256, 256, 16)  #256 in
        self.up2 = UpSampleBlock(128, 128, 8)  #128 in
        self.up3 = UpSampleBlock(64, 64, 4)    #64 in
        self.up4 = UpSampleBlock(64, 64, 2)   #64 in  
        nf_up = 256+128+64+64+3
        nf_mid = 256  
 
        mid_layers = []
        mid_layers += [ConvBlock(nf_up,nf_mid, bn=True, actn=False)]
        
        for i in range(8): 
            mid_layers.append(
                ResSequential(
                [ConvBlock(nf_mid, nf_mid, actn=True, bn=False), 
                 ConvBlock(nf_mid, nf_mid, actn=False, bn=False)], 0.1)
            )
            
        mid_layers += [ConvBlock(nf_mid,nf_mid, actn=False), 
                       ConvBlock(nf_mid, 3, bn=False, actn=False)]
        self.upconv = nn.Sequential(*mid_layers)
             
        out_layers = []
        out_layers += [ConvBlock(6, 3, ks=1, bn=False, actn=False)]
        self.out = nn.Sequential(*out_layers)
        
    def forward(self, x): 
        self.rn(x)
        x1 = self.up1(self.sfs[3].features)
        x2 = self.up2(self.sfs[2].features)
        x3 = self.up3(self.sfs[1].features)
        x4 = self.up4(self.sfs[0].features) 
        x5 = self.upconv(torch.cat([x, x1, x2, x3, x4], dim=1))
        return F.tanh(self.out(torch.cat([x, x5], dim=1)))    
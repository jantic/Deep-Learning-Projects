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

class DCCritic(CriticModule):
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

    def get_layer_groups(self)->[]:
        return children(self)
    
    def forward(self, x):
        x=self.initial(x)
        x=self.mid(x)
        return self.out(x)


class FeatureCritic(CriticModule):           
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

    def get_layer_groups(self)->[]:
        lgs = list(split_by_idxs(children(self.rn), [self.lr_cut]))
        return lgs + [children(self)[1:]]

class WGANGenTrainingResult():
    def __init__(self, gcost: np.array, gcount: int):
        self.gcost=gcost
        self.gcount=gcount

class WGANCriticTrainingResult():
    def __init__(self, wdist: np.array, gpenalty: np.array, dreal: np.array, dfake: np.array, dcost: np.array):
        self.wdist=wdist
        self.gpenalty=gpenalty
        self.dreal=dreal
        self.dfake=dfake
        self.dcost=dcost

class WGANTrainer():
    def __init__(self, netD: nn.Module, netG: GeneratorModule, md: ImageData, bs:int, sz:int, dpath: Path, gpath: Path):
        self.netD = netD
        self.netG = netG
        self.md = md
        self.bs = bs
        self.sz = sz
        self.dpath = dpath
        self.gpath = gpath
        self._train_loop_hooks = OrderedDict()

    def register_train_loop_hook(self, hook):
        handle = hooks.RemovableHandle(self._train_loop_hooks)
        self._train_loop_hooks[handle.id] = hook
        return handle

    def train(self, lrs_critic:[int], lrs_gen:[int], clr_critic: (int)=(20,10), clr_gen: (int)=(20,10), 
            cycle_len:int =1, epochs: int=1, first:bool=True):

        self.gen_sched = self._generate_clr_sched(self.netG, clr_gen, lrs_gen, cycle_len)
        self.critic_sched = self._generate_clr_sched(self.netD, clr_critic, lrs_critic, cycle_len)

        gcount = 0
        self.critic_sched.on_train_begin()
        self.gen_sched.on_train_begin()

        for epoch in trange(epochs):
            gcount = self._train_one_epoch(gcount, first)

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
    
    def _train_one_epoch(self, gcount: int, first: bool)->int:
        self.netD.train()
        self.netG.train()
        data_iter = iter(self.md.trn_dl)
        n = len(self.md.trn_dl)
        with tqdm(total=n) as pbar:
            while True:
                cresult = self._train_critic(first, gcount, data_iter, pbar)
                
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

    def _get_num_critic_iters(self, first: bool, gcount: int)->int:
        return 100 if (first and (gcount < 25) or (gcount % 500 == 0)) else 5

    def _get_next_training_images(self, data_iter: Iterable)->(torch.Tensor,torch.Tensor):
        x, y = next(data_iter, (None, None))
        if x is None:
            return (None, None)
        orig_image = V(x)
        real_image = V(y) 
        return (orig_image, real_image)

    def _determine_abs_noise_loss(self, orig_image):
        return abs(self._get_raw_noise_loss(orig_image))

    def _normalize_loss(self, orig_image: torch.Tensor, loss: torch.Tensor):
        abs_noise_loss = self._determine_abs_noise_loss(orig_image)
        return (loss/abs_noise_loss)*10

    def _calculate_wdist(self, orig_image: torch.Tensor, real_image: torch.Tensor, fake_image: torch.Tensor)->torch.Tensor:
        dreal = self._get_dscore(real_image, orig_image)
        dfake = self._get_dscore(V(fake_image.data), orig_image)
        wdist = dfake - dreal
        return wdist, dfake, dreal

    def _is_equilibrium(self, cresult: WGANCriticTrainingResult):
        dreal  = cresult.dreal
        dfake = cresult.dfake

        if dreal < dfake:
            return False
        return abs(dreal + dfake) < (abs(dreal) + abs(dfake))*0.30

    def _train_critic(self, first: bool, gcount: int, data_iter: Iterable, pbar: tqdm)->WGANCriticTrainingResult:
        self.netD.set_trainable(True)
        self.netG.set_trainable(False)
        j = 0
        #d_iters = self._get_num_critic_iters(first, gcount)
        cresult=None
        equilibrium = False

        while not equilibrium:
            orig_image, real_image = self._get_next_training_images(data_iter)
            if orig_image is None:
                return cresult
            j += 1
            #To help boost critic early on, we're doing both noise and generator based training, since
            #the generator never actually starts by creating noise
            #self._train_critic_once(orig_image, real_image, noise_gen=True)
            cresult = self._train_critic_once(orig_image, real_image, noise_gen=False)
            pbar.update()
            equilibrium = self._is_equilibrium(cresult)
        
        return cresult

    def _train_critic_once(self, orig_image: torch.Tensor, real_image: torch.Tensor, noise_gen:bool= False)->WGANCriticTrainingResult:                     
        #Higher == Real
        fake_image = self._create_noise_batch() if noise_gen else self.netG(orig_image)
        wdist, dfake, dreal = self._calculate_wdist(orig_image, real_image, fake_image)
        self.netD.zero_grad()        
        gpenalty = self._calc_gradient_penalty(real_image.data, fake_image.data, orig_image)              
        dcost = dfake - dreal + gpenalty
        dcost.backward()
        self.critic_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(dcost))
        self.gen_sched.on_batch_end(to_np(-dfake))
        return WGANCriticTrainingResult(to_np(wdist), to_np(gpenalty), to_np(dreal), to_np(dfake), to_np(dcost))
    
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
        gcost  = -self._get_dscore(fake_image, orig_image)
        gcost.backward()
        self.gen_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(cresult.dcost))
        self.gen_sched.on_batch_end(to_np(gcost))
        return WGANGenTrainingResult(to_np(gcost), gcount)

    def _save_if_applicable(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        if cresult is None or gresult is None:
            return

        if gresult.gcount % 100 == 0:
            save_model(self.netD, self.dpath)
            save_model(self.netG, self.gpath)

    def _get_dscore(self, new_image: torch.Tensor, orig_image: torch.Tensor):
        #return self._normalize_loss(orig_image, self.netD(new_image, orig_image))
        return self.netD(new_image, orig_image)

    def _calc_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor, orig_data: torch.Tensor)->torch.Tensor:
        lamda = 10 # Gradient penalty lambda hyperparameter
        alpha = torch.rand(self.bs, 1)
        alpha = alpha.expand(self.bs, real_data.nelement()//self.bs).contiguous().view(self.bs, 3, self.sz, self.sz)
        alpha = alpha.cuda()
        interpolates = alpha*real_data + (1-alpha)*fake_data
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self._get_dscore(interpolates, orig_data)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamda
        return gradient_penalty
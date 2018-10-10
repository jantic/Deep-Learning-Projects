from fasterai.modules import *
from fasterai.generators import *
from fasterai.loss import *
from torch import autograd
from collections import Iterable
import torch.utils.hooks as hooks
from torch.nn.utils.spectral_norm import spectral_norm


class CriticModule(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    def freeze_to(self, n:int):
        c=self.get_layer_groups()
        for l in c:     set_trainable(l, False)
        for l in c[n:]: set_trainable(l, True)
    
    def set_trainable(self, trainable:bool):
        set_trainable(self, trainable)

    @abstractmethod
    def get_layer_groups(self)->[]:
        pass

class ResCritic(CriticModule): 
    def _generate_eval_layers(self, ni:int, nf:int, scale:int):
        layers = [] 
        cndf = nf
        
        layers.append(ConvBlock(ni, cndf, 7, 1, bn=False))
        layers.append(DownSampleResBlock(ni=cndf, nf=cndf*2, dropout=0.2, bn=False))
        cndf = cndf*2      
        layers.append(ResBlock(nf=cndf, ks=3, bn=False))

        scale_count = 0
        for i in range(int(math.log(scale,2))-1):
            layers.append(DownSampleResBlock(ni=cndf, nf=cndf*2, bn=False))
            cndf = int(cndf*2)


        return nn.Sequential(*layers), cndf
            
    def __init__(self, nf:int=64, scale:int=32):
        super().__init__()
        assert (math.log(scale,2)).is_integer()
        self.pixel_eval, nf_mid = self._generate_eval_layers(3, nf, scale)
        self.mid = ResBlock(nf=nf_mid, ks=3, bn=False) 
        self.out = nn.Conv2d(nf_mid, 1, kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, input: torch.Tensor):
        p = self.pixel_eval(input)
        before_last = self.mid(p)
        x = self.out(before_last)
        return x, before_last

    def get_layer_groups(self)->[]:
        return [children(self)]


class DCCritic(CriticModule):

    def _generate_reduce_layers(self, nf:int):
        layers=[]
        layers.append(ConvBlock(nf, nf*2, 4, 2, bn=False))
        layers.append(nn.Dropout2d(0.5))
        return layers

    def __init__(self, ni:int, nf:int, scale:int=32):
        super().__init__()

        assert (math.log(scale,2)).is_integer()
        self.initial = nn.Sequential(
            ConvBlock(ni, nf, 4, 2, bn=False),
            nn.Dropout2d(0.2))

        cndf = nf
        mid_layers =  []  
        mid_layers.append(ConvBlock(cndf, cndf, 3, 1, bn=False))
        mid_layers.append(nn.Dropout2d(0.5))
        scale_count = 0

        for i in range(int(math.log(scale,2))-1):
            layers = self._generate_reduce_layers(nf=cndf)
            mid_layers.extend(layers)
            cndf = int(cndf*2)
   
        self.mid = nn.Sequential(*mid_layers)
        self.prefinal = nn.Sequential(*self._generate_reduce_layers(nf=cndf))
        cndf = int(cndf*2)

        out_layers=[]
        out_layers.append(nn.Conv2d(cndf, 1, kernel_size=1, stride=1, padding=0, bias=False))   
        self.out = nn.Sequential(*out_layers) 

    def get_layer_groups(self)->[]:
        return children(self)
    
    def forward(self, input):
        x=self.initial(input)
        x=self.mid(x)
        before_last = self.prefinal(x)
        return self.out(before_last), before_last

class WGANGenTrainingResult():
    def __init__(self, gcost: np.array, gcount: int, gaddlloss: np.array):
        self.gcost=gcost
        self.gcount=gcount
        self.gaddlloss=gaddlloss

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
            bs:int, sz:int, dpath: Path, gpath: Path, gplambda=10, citers=1, 
            save_iters=1000, genloss_fns:[]=[]):
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
        self._train_begin_hooks = OrderedDict()
        self.genloss_fns = genloss_fns
        self.save_iters=save_iters

    def register_train_loop_hook(self, hook):
        handle = hooks.RemovableHandle(self._train_loop_hooks)
        self._train_loop_hooks[handle.id] = hook
        return handle

    def register_train_begin_hook(self, hook):
        handle = hooks.RemovableHandle(self._train_begin_hooks)
        self._train_begin_hooks[handle.id] = hook
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

    def _get_raw_noise_loss(self):
        noise_image = self._create_noise_batch()
        return self.netD(noise_image)

    def _create_noise_batch(self): 
        raw_random = V(torch.randn(self.bs, 3, self.sz,self.sz).normal_(0, 1))
        return F.tanh(raw_random)

    def _generate_clr_sched(self, model:nn.Module, use_clr_beta: (int), lrs: [int], cycle_len: int):
        wds = 1e-7
        opt_fn = partial(optim.Adam, betas=(0.5, 0.9))
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
        self._call_train_begin_hooks()

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
    
    def _call_train_begin_hooks(self):
        for hook in self._train_begin_hooks.values():
            hook_result = hook()
            if hook_result is not None:
                raise RuntimeError(
                    "train begin hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))

    def _call_train_loop_hooks(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        for hook in self._train_loop_hooks.values():
            hook_result = hook(gresult, cresult)
            if hook_result is not None:
                raise RuntimeError(
                    "train loop hooks should never return any values, but '{}'"
                    "didn't return None".format(hook))

    def _get_next_training_images(self, data_iter: Iterable)->(torch.Tensor,torch.Tensor):
        x, y = next(data_iter, (None, None))
        if x is None:
            return (None, None)
        orig_image = V(x)
        real_image = V(y) 
        return (orig_image, real_image)


    def _calculate_wdist(self, real_image: torch.Tensor, fake_image: torch.Tensor)->torch.Tensor:
        dreal = self._get_dscore(real_image)
        dfake = self._get_dscore(V(fake_image.data))
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
        wdist, dfake, dreal = self._calculate_wdist(real_image, fake_image)
        self.netD.zero_grad()        
        gpenalty = self._calc_gradient_penalty(real_image.data, fake_image.data)     
        conpenalty = self._consistency_penalty(real_image.data)      
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
        gcost = -self._get_dscore(fake_image)
        gaddlloss = self._calc_addl_gen_loss(real_image, fake_image) 
        total_loss = gcost if gaddlloss is None else gcost + gaddlloss
        total_loss.backward()
        self.gen_sched.layer_opt.opt.step()
        self.critic_sched.on_batch_end(to_np(cresult.dcost))
        self.gen_sched.on_batch_end(to_np(gcost))
        return WGANGenTrainingResult(to_np(gcost), gcount, to_np(gaddlloss))

    def _save_if_applicable(self, gresult: WGANGenTrainingResult, cresult: WGANCriticTrainingResult):
        if cresult is None or gresult is None:
            return

        if gresult.gcount % self.save_iters == 0:
            save_model(self.netD, self.dpath)
            save_model(self.netG, self.gpath)

    def _get_dscore(self, new_image: torch.Tensor):
        final, _ = self.netD(new_image)
        return final.mean()

    def _calc_addl_gen_loss(self, real_data: torch.Tensor, fake_data: torch.Tensor)->torch.Tensor:
        total_loss = None
        for loss_fn in self.genloss_fns:
            loss = loss_fn(fake_data, real_data)
            total_loss = loss if total_loss is None else total_loss + loss
        return total_loss


    def _calc_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor)->torch.Tensor:
        alpha = torch.rand(self.bs, 1)
        alpha = alpha.expand(self.bs, real_data.nelement()//self.bs).contiguous().view(self.bs, 3, self.sz, self.sz)
        alpha = alpha.cuda()
        differences = fake_data - real_data
        interpolates = real_data + (alpha*differences)
        interpolates = interpolates.cuda()
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        disc_interpolates = self._get_dscore(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.gplambda
        return gradient_penalty

    def _consistency_penalty(self, real_data: torch.Tensor):
        d1, d_1 = self.netD(real_data)
        d2, d_2 = self.netD(real_data)
        consistency_term = (d1 - d2).norm(2, dim=1) + 0.1 * (d_1 - d_2).norm(2, dim=1)
        return consistency_term.mean()
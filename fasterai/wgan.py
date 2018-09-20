from fasterai.modules import *
from fasterai.visualize import *
from torch import autograd

class DCCritic(nn.Module):
    def __init__(self, ni:int, nf:int, sz:int):
        super().__init__()
        layers = [] 
        layers.append(ConvBlock(ni, nf, 4, 2, bn=False))
        csize,cndf = sz//2,nf
        layers.append(nn.LayerNorm([cndf, csize, csize]))
        layers.append(ConvBlock(cndf, cndf, 3, 1, bn=False))
        layers.append(nn.LayerNorm([cndf, csize, csize]))

        while csize > 8:
            layers.append(ConvBlock(cndf, cndf*2, 4, 2, bn=False))
            cndf = int(cndf*2)
            csize = int(csize//2)
            layers.append(nn.LayerNorm([cndf, csize, csize]))
        
        layers.append(nn.Conv2d(cndf, 1, 4, padding=0, bias=False))    
        self.seq = nn.Sequential(*layers) 
    
    def forward(self, x):
        return self.seq(x)


class FeatureCritic(nn.Module):
    def set_trainable(self, trainable):
        set_trainable(self, trainable)
        set_trainable(self.rn, False)
                
    def __init__(self, sz:int):
        super().__init__()        
        self.rn, self.lr_cut = get_pretrained_resnet_base()
        set_trainable(self.rn, False)
        self.sfs = [SaveFeatures(self.rn[i]) for i in [2,4,5,6]]      
        self.feature_eval_1 = DCCritic(256*2, 256, sz//16)
        self.feature_eval_2 = DCCritic(128*2, 128, sz//8)
        self.feature_eval_3 = DCCritic(64*2, 64, sz//4)
        self.feature_eval_4 = DCCritic(64*2, 64, sz//2)     
        self.pixel_eval = DCCritic(6, 64, sz)
        
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
        return f1.mean() + f2.mean() + f3.mean()  + f4.mean() + p.mean()


class WGANTrainer():

    def __init__(self, netD: nn.Module, netG: nn.Module, md, bs:int, sz:int, dpath: Path, gpath: Path, lr:float=1e-4):
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

    def train(self, niter: int=1):
        gen_iterations = 0
        for epoch in trange(niter):
            self.netD.train()
            self.netG.train()
            data_iter = iter(self.md.trn_dl)
            i,n = 0,len(self.md.trn_dl)
            n = n-(n%self.bs)
            with tqdm(total=n) as pbar:
                while i < n:
                    self.netD.set_trainable(True)
                    self.netG.set_trainable(False)
                    j = 0
                    equilibrium = False
                    while (not equilibrium) and (i < n) and j<10000:
                        j += 1; i += 1
                        #or p in netD.parameters(): p.data.clamp_(-0.01, 0.01)
                        x, y = next(data_iter)
                        orig_image = V(x)
                        real_image = V(y)                        
                        #Higher == Real
                        disc_real = self.netD(real_image, orig_image)
                        fake_image = self.netG(orig_image)
                        disc_fake = self.netD(V(fake_image.data), orig_image)
                        equilibrium = self._is_equilibrium(disc_real, disc_fake)
                        
                        self.netD.zero_grad()
                            
                        gradient_penalty = self._calc_gradient_penalty(real_image.data, fake_image.data, orig_image)              
                        disc_cost = disc_fake - disc_real + gradient_penalty
                        w_dist = disc_fake - disc_real
                        disc_cost.backward()
                        self.optimizerD.step()
                        pbar.update()
                
                        self._progress_update(i, w_dist, gradient_penalty, disc_real, disc_fake, gen_iterations)

                        
                    self.netD.set_trainable(False)
                    self.netG.set_trainable(True)
                    self.netG.zero_grad()
                    
                    x, y = next(data_iter)
                    orig_image = V(x)
                    real_image = V(y)   
                    fake_image = self.netG(orig_image)
                    gen_cost  = -self.netD(fake_image, orig_image)
                    gen_cost.backward()
                    self.optimizerG.step()
                    gen_iterations += 1
                    
                    self._progress_update(i, w_dist, gradient_penalty, disc_real, disc_fake, gen_iterations)

    def _is_equilibrium(self, disc_real, disc_fake):
        if disc_real < disc_fake:
            return False
            
        return abs(disc_real + disc_fake) < (abs(disc_real) + abs(disc_fake))*0.30   

    def _progress_update(self, i, w_dist, gradient_penalty, disc_real, disc_fake, ecount):
        if i % 50 == 0:
            print(f'\nWDist {to_np(w_dist)}; GPenalty {to_np(gradient_penalty)}; RScore {to_np(disc_real)};'
            + f' FScore {to_np(disc_fake)}; ECount: {ecount}')

        if i % 500 == 0:
            visualize_image_gen_model(self.md, self.netG, 500, 8)
            save_model(self.netD, self.dpath)
            save_model(self.netG, self.gpath)

    def _calc_gradient_penalty(self, real_data, fake_data, orig_data):
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
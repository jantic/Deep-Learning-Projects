import multiprocessing
from torch import autograd
from fastai.conv_learner import *
from fastai.transforms import TfmType
from fasterai.transforms import *
from fasterai.images import *
from fasterai.dataset import *
from fasterai.visualize import *
from fasterai.callbacks import *
from fasterai.loss import *
from fasterai.modules import *
from fasterai.wgan import *
from fasterai.generators import *
from fastai.torch_imports import *
from pathlib import Path
from itertools import repeat
import tensorboardX
torch.cuda.set_device(1)
plt.style.use('dark_background')
torch.backends.cudnn.benchmark=True

IMAGENET = Path('/media/jason/Projects/Deep Learning/data/dogscats/sample/train')
CIFAR10 = Path('/media/jason/Projects/Deep Learning/data/cifar10/train')

proj_id = 'bw2color_WganUnetSched6'
TENSORBOARD_PATH = Path('/media/jason/Projects/Deep Learning/data/tensorboard/' + proj_id)

c_lr=1e-3
c_lrs = np.array([c_lr,c_lr,c_lr])

g_lr=c_lr/3
g_lrs = np.array([g_lr/10000,g_lr/100,g_lr])

x_tfms = [BlackAndWhiteTransform()]
torch.backends.cudnn.benchmark=True



netG = Unet34(nf_factor=1, self_attention=False).cuda()
netGVis = ModelVisualizationHook(TENSORBOARD_PATH, netG, 'netG')
#load_model(netG, gpath)

netD = DCCritic(ni=3, nf=256, scale=32, self_attention=False).cuda()
netDVis = ModelVisualizationHook(TENSORBOARD_PATH, netD, 'netD')
#load_model(netD, dpath)
trainer = WGANTrainer(netD=netD, netG=netG, genloss_fns=[FeatureLoss(multiplier=1e5)])
trainerVis = WganVisualizationHook(TENSORBOARD_PATH, trainer, 'trainer')

scheds=[]
#scheds.extend(WGANTrainSchedule.generate_schedules(szs=[64,128,256], bss=[64,16,4], path=IMAGENET, x_tfms=x_tfms, keep_pcts=[0.5,0.2,0.1], save_base_name=proj_id, 
        #critic_lrs_start=c_lrs, gen_lrs_start=g_lrs, lrs_change_factor=0.5, random_seed=42))
scheds.extend(WGANTrainSchedule.generate_schedules(szs=[64,128,256], bss=[4,4,4], path=IMAGENET, x_tfms=x_tfms, keep_pcts=[1.0,1.0,1.0], save_base_name=proj_id, 
        critic_lrs_start=c_lrs, gen_lrs_start=g_lrs, lrs_change_factor=0.5, random_seed=42))

netG.freeze_to(1)
trainer.train(scheds=scheds)
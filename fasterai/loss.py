
from fastai.conv_learner import *
from fasterai.modules import *
import torchvision.models as models

class FeatureLoss(nn.Module):
    def __init__(self, block_wgts: [float] = [0.65,0.25,0.1], multiplier:float=1.0):
        super().__init__()
        m_vgg = vgg16(True)
        
        blocks = [i-1 for i,o in enumerate(children(m_vgg))
              if isinstance(o,nn.MaxPool2d)]
        blocks, [m_vgg[i] for i in blocks]
        layer_ids = blocks[:3]
        
        vgg_layers = children(m_vgg)[:23]
        m_vgg = nn.Sequential(*vgg_layers).cuda().eval()
        set_trainable(m_vgg, False)
        
        self.m,self.wgts = m_vgg,block_wgts
        self.sfs = [SaveFeatures(m_vgg[i]) for i in layer_ids]
        self.multiplier = multiplier

    def forward(self, input, target, sum_layers=True):
        self.m(VV(target.data))
        res = [F.l1_loss(input,target)/100]
        targ_feat = [V(o.features.data.clone()) for o in self.sfs]
        self.m(input)
        res += [F.l1_loss(self._flatten(inp.features),self._flatten(targ))*wgt
               for inp,targ,wgt in zip(self.sfs, targ_feat, self.wgts)]
        if sum_layers: res = sum(res)
        return res*self.multiplier
    
    def _flatten(self, x): 
        return x.view(x.size(0), -1)
    
    def close(self):
        for o in self.sfs: o.remove()


class PerceptualLoss(nn.Module):		
    def __init__(self, multiplier:float=1.0):
        super().__init__()
        self.loss_fn = nn.MSELoss()
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i,layer in enumerate(list(cnn)):
            model.add_module(str(i),layer)
            if i == conv_3_3_layer: 
                break
        self.model = model
        self.multiplier = multiplier
	
    def forward(self, input, target):
        fake_features = self.model.forward(input)
        real_features = self.model.forward(target)
        loss = self.loss_fn(fake_features, real_features.detach())
        return loss*self.multiplier

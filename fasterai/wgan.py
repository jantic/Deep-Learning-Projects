from fasterai.modules import *

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
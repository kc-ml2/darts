#!/usr/bin/env python
# coding: utf-8



import  torch
from    torch import nn
import  torch.nn.functional as F
from    models.operations import OPS, FactorizedReduce, ReLUConvBN
from    models.genotypes import PRIMITIVES, Genotype



# In[3]:


class MixedLayer(nn.Module):
    
    def __init__(self, c, stride):
        
        super(MixedLayer, self).__init__()
        
        self.layers = nn.ModuleList()
        
        for primitive in PRIMITIVES:
            layer = OPS[primitive](c, stride, False)
        
            if 'pool' in primitive:
                layer = nn.Sequential(layer, nn.BatchNorm2d(c, affine=False))
                
            self.layers.append(layer)
            
    def forward(self, x, weights):
        
        res = [w * layer(x) for w,layer in zip(weights, self.layers)]
#         print('======================================================')
#         print("forwards debug : ",[ts.shape for ts in res])
        res = sum(res)
        
        return res


# In[16]:


class Cell(nn.Module):
    
    def __init__(self, steps, multiplier, cpp, cp, c, reduction, reduction_prev):
        super(Cell, self).__init__()
        
        self.reduction = reduction
        self.reduction_prev = reduction_prev
        
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(cpp, c, affine=False)
        else :
            self.preprocess0 = ReLUConvBN(cpp, c, 1, 1, 0, affine=False)
            
        self.preprocess1 = ReLUConvBN(cp, c, 1,1,0,affine=False)
        
        self.steps=steps
        self.multiplier = multiplier
        
        self.layers = nn.ModuleList()
        
        for i in range(self.steps):
            for j in range(2+i):
                stride = 2 if reduction and j < 2 else 1
                layer = MixedLayer(c, stride)
                self.layers.append(layer)
                
    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        
#         print("Cell forwards!")
        
        states = [s0, s1]
        offset = 0
        
        for i in range(self.steps):
            s = sum(self.layers[offset + j](h, weights[offset+j]) for j, h in enumerate(states))
            offset += len(states)
            
            states.append(s)
            
        return torch.cat(states[-self.multiplier:], dim=1)
        


# In[17]:


class Network(nn.Module):
    
    def __init__(self, c, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        
        super(Network, self).__init__()
        
        self.c = c
        self.num_classes = num_classes
        self.layers = layers
        self.criterion = criterion
        self.steps = steps
        self.multiplier = multiplier
        
        c_curr = stem_multiplier *c
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, c_curr, 3, padding=1, bias = False),
            nn.BatchNorm2d(c_curr)
        )
        
        cpp, cp, c_curr = c_curr, c_curr, c
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction= True
            else:
                reduction= False
                
            cell = Cell(steps, multiplier, cpp, cp, c_curr, reduction, reduction_prev)
            
            reduction_prev = reduction
            
            self.cells += [cell]
            
            cpp, cp = cp, multiplier * c_curr
            
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Linear(cp, num_classes)
        
        k = sum(1 for i in range(self.steps) for j in range(2+i))
        num_ops = len(PRIMITIVES)
            
        self.alpha_normal = nn.Parameter(torch.randn(k, num_ops))
        self.alpha_reduce = nn.Parameter(torch.randn(k, num_ops))
        with torch.no_grad():
            self.alpha_normal.mul_(1e-3)
            self.alpha_reduce.mul_(1e-3)
        self._arch_parameters = [
            self.alpha_normal,
            self.alpha_reduce,
        ]
    
    
    def new(self):
        model_new = Network(self.c, self.num_classes, self.layers, self.criterion).cuda()
        for x,y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    
    
    def forward(self, x):
        s0 = s1 = self.stem(x)
#         print("Network forwards!")
        
        for i, cell in enumerate(self.cells):
            
            if cell.reduction:
                weights = F.softmax(self.alpha_reduce, dim=-1)
            else:
                weights = F.softmax(self.alpha_normal, dim=-1)
            
            s0, s1 = s1, cell(s0, s1, weights)
            
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        
        return logits
    
    
    def loss(self, x, target):
        logits = self(x)
        return self.criterion(logits, target)
    
    
    def arch_parameters(self):
        return self._arch_parameters
    
    
    def genotype(self):
        
        def _parse(weights):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start+n
                W = weights[start:end].copy()
                edges = sorted(range(i+2),
                              key=lambda x: -max(W[x][k]
                                                    for k in range(len(W[x]))
                                                    if k != PRIMITIVES.index('none'))
                              )[:2]
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best],j))
                start = end
                n += 1
            return gene
        
        gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())
        
        concat = range(2+self.steps - self.multiplier, self.steps +2)
        genotype = Genotype(
            normal = gene_normal, normal_concat=concat,
            reduce = gene_reduce, reduce_concat=concat
        )
        
        return genotype


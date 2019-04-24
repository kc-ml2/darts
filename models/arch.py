#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from torch import optim, autograd


# In[2]:


def concat(xs):
    return torch.cat([x.view(-1) for x in xs])


# In[3]:


class Arch:
    
    def __init__(self, model, args):
        self.momentum = args.momentum
        self.wd = args.wd
        self.model = model
        
        self.optimizer = optim.Adam(self.model.arch_parameters(),
                                    lr=args.arch_lr,
                                    betas=(0.5, 0.999),
                                    weight_decay=args.arch_wd
                                   )
    
    
    def comp_unrolled_model(self, x, target, eta, optimizer):
        loss = self.model.loss(x, target)
        theta = concat(self.model.parameters()).detach()
        
        try:
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except:
            moment = torch.zeros_like(theta)
            
        dtheta = concat(autograd.grad(loss, self.model.parameters())).data
        
        theta = theta.sub(eta, moment+dtheta+self.wd*theta)
        
        unrolled_model = self.construct_model_from_theta(theta)
        
        return unrolled_model
    
    
    def step(self, x_train, target_train, x_valid, target_valid, eta, optimizer, unrolled):
        self.optimizer.zero_grad()
        
        if unrolled:
            self.backward_step_unrolled(x_train, target_train, x_valid, target_valid, eta, optimizer)
        else:
            self.backward_step(x_valid, target_valid)
            
        self.optimizer.step()
        
        
    def backward_step(self, x_valid, target_valid):
        loss = self.model.loss(x_valid, target_valid)
        
        loss.backward()
        
        
    def backward_step_unrolled(self, x_train, target_train, x_valid, target_valid, eta, optimizer):
        unrolled_model = self.comp_unrolled_model(x_train, target_train, eta, optimizer)
        unrolled_loss = unrolled_model.loss(x_valid, target_valid)
        
        unrolled_loss.backward()
        
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad.data for v in unrolled_model.parameters()]
        implicit_grads = self.hessian_vector_product(vector, x_train, target_train)
        
        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
            
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = g.data
            else:
                v.grad.data.copy_(g.data)
    
    
    def construct_model_from_theta(self, theta):
        model_new = self.model.new()
        model_dict = self.model.state_dict()
        
        params, offset = {}, 0
        for k,v in self.model.named_parameters():
            v_length = v.numel()
            params[k] = theta[offset: offset + v_length].view(v.size())
            offset += v_length
        
        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()
    
    def hessian_vector_product(self, vector, x, target, r=1e-2):
        
        R = r / concat(vector).norm()
        
        for p, v in zip(self.model.parameters(),vector):
            p.data.add_(R, v)
        
        loss = self.model.loss(x, target)
        grads_p = autograd.grad(loss, self.model.arch_parameters())
        
        
        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2*R, v)
        
        loss = self.model.loss(x, target)
        grads_n = autograd.grad(loss, self.model.arch_parameters())
        
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
        
        h = [(x - y).div_(2*R) for x,y in zip(grads_p, grads_n)]
        
        return h
        


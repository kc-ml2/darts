# coding: utf-8
import copy
import torch




class Architect():
    
    def __init__(self, net, w_momentum, w_weight_decay):
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        
        
    def virtual_step(self, trn_X, trn_y, w_lr, w_optim):
        """
        compute unrolled weight w' (virtual step)
        
        step process:
        1) forward
        2) calc loss
        3) compute gradient (backprop)
        4) update gradient
        
        Args:
            w_lr : learning rate for virtual gradient step (same as weight lr)
            w_optim : weights optimizer
        """
        
        # forward & calc
        loss = self.net.loss(trn_X, trn_y)
        
        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())
        
        with torch.no_grad():
            
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m=w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - w_lr*(m+g+self.w_weight_decay*w))
                
            for a, va, in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)
                
                
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, w_lr, w_optim):
        
        self.virtual_step(trn_X, trn_y, w_lr, w_optim)
        
        loss = self.v_net.loss(val_X, val_y)
        
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]
        
        hessian = self.compute_hessian(dw,trn_X, trn_y)
        
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - w_lr * h
    
    
    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm
        
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p+= eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas())
        
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas())
        
        with torch.no_grad():
            for p,d in zip(self.net.weights(), dw):
                p += eps * d
                
        hessian = [(p-n) / 2.*eps for p,n in zip(dalpha_pos, dalpha_neg)]
        return hessian
        
        







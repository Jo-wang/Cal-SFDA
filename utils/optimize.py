import torch 
import torch.nn as nn

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, len_loader, args):
    lr = lr_poly(args.learning_rate, i_iter, args.epochs*len_loader, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


import math
import numpy as np

class ProportionScheduler:
    def __init__(self, pytorch_lr_scheduler, max_lr, min_lr, max_value, min_value):
        self.t = 0    
        self.pytorch_lr_scheduler = pytorch_lr_scheduler
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_value = max_value
        self.min_value = min_value
        
        assert (max_lr > min_lr) or ((max_lr==min_lr) and (max_value==min_value)), "Current scheduler for `value` is scheduled to evolve proportionally to `lr`," \
        "e.g. `(lr - min_lr) / (max_lr - min_lr) = (value - min_value) / (max_value - min_value)`. Please check `max_lr >= min_lr` and `max_value >= min_value`;" \
        "if `max_lr==min_lr` hence `lr` is constant with step, please set 'max_value == min_value' so 'value' is constant with step."
    
        assert max_value >= min_value
        
        self.step() 
    
    def lr(self):
        return self._last_lr[0]
                
    def step(self):
        self.t += 1
        if hasattr(self.pytorch_lr_scheduler, "_last_lr"):
            lr = self.pytorch_lr_scheduler._last_lr[0]
        else:
            lr = self.pytorch_lr_scheduler.optimizer.param_groups[0]['lr']
            
        if self.max_lr > self.min_lr:
            value = self.min_value + (self.max_value - self.min_value) * (lr - self.min_lr) / (self.max_lr - self.min_lr)
        else:
            value = self.max_value
        
        self._last_lr = [value]
        return value
        
class SchedulerBase:
    def __init__(self, T_max, max_value, min_value=0.0, init_value=0.0, warmup_steps=0, optimizer=None):
        super(SchedulerBase, self).__init__()
        self.t = 0
        self.min_value = min_value
        self.max_value = max_value
        self.init_value = init_value
        self.warmup_steps = warmup_steps
        self.total_steps = T_max    
        self._last_lr = [init_value]              
        self.optimizer = optimizer

    def step(self):
        if self.t < self.warmup_steps:
            value = self.init_value + (self.max_value - self.init_value) * self.t / self.warmup_steps
        elif self.t == self.warmup_steps:
            value = self.max_value
        else:
            value = self.step_func()
        self.t += 1


        if self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = value
                
        self._last_lr = [value]
        return value

    def step_func(self):
        pass
    
    def lr(self):
        return self._last_lr[0]

class LinearScheduler(SchedulerBase):
    def step_func(self):
        value = self.max_value + (self.min_value - self.max_value) * (self.t - self.warmup_steps) / (
                    self.total_steps - self.warmup_steps)
        return value

class CosineScheduler(SchedulerBase):
    def step_func(self):
        phase = (self.t-self.warmup_steps) / (self.total_steps-self.warmup_steps) * math.pi
        value = self.min_value + (self.max_value-self.min_value) * (np.cos(phase) + 1.) / 2.0
        return value

class PolyScheduler(SchedulerBase):
    def __init__(self, poly_order=-0.5, *args, **kwargs):
        super(PolyScheduler, self).__init__(*args, **kwargs)
        self.poly_order = poly_order
        assert poly_order<=0, "Please check poly_order<=0 so that the scheduler decreases with steps"

    def step_func(self):
        value = self.min_value + (self.max_value-self.min_value) * (self.t - self.warmup_steps)**self.poly_order
        return value



def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, len_loader, args):
    lr = lr_poly(args.learning_rate, i_iter, args.epochs*len_loader, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr
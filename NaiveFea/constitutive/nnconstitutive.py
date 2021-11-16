import numpy as np
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian

class NNMaterial(nn.Module):
    """common material define by neural network"""
    def __init__(self):
        super().__init__()
        self.dtype=torch.float32

    def __call__(self, strain):
        if type(strain)!=torch.Tensor:
            strain=nn.Parameter(torch.tensor(strain,dtype=self.dtype))
        self.strain=strain
        return self.forward(strain).detach().numpy()

    def cal_Jacobian(self):
        self.Jacobian=jacobian(self.forward,self.strain)

class NNLinearMaterial(NNMaterial):
    """constitutive relationship of isotropic linear material"""
    def __init__(self,E,nv):
        super().__init__()
        self.D=nn.Parameter(self.__cal_D(E,nv),requires_grad=False)
    
    def __cal_D(self,E,nv):
        return E/((1.0+nv)*(1.0-2*nv))*torch.tensor([
            [1.0-nv,nv,0.0], 
            [nv,1.0-nv,0.0], 
            [0.0,0.0,0.5-nv]],dtype=self.dtype)
    
    def forward(self,strain):
        self.stress=F.linear(strain,self.D)
        self.cal_jacobian()
        return self.stress

    def cal_jacobian(self):
        self.Jacobian=self.D
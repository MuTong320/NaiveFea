import numpy as np

class CommonMaterial:
    def __call__(self, variable):
        return self.forward(variable)

    def __init__(self):
        self.name='Unnamed material'
    
    def set_name(self,name):
        self.name=name
    
    def forward(self,variable):
        return np.zeros_like(variable)

class LinearMaterial(CommonMaterial):
    """constitutive relationship of isotropic linear material.
    E: Young's modulus;
    nv: Poisson's ratio."""
    def __init__(self,E,nv):
        super().__init__()
        self.E=E 
        self.nv=nv 
        self.cal_D()

    def cal_D(self):
        E=self.E
        nv=self.nv
        self.D=E/(1.0-nv**2)*np.array([
            [1.0,nv,0.0], 
            [nv,1.0,0.0], 
            [0.0,0.0,0.5*(1-nv)]])

    def forward(self,strain,autograd=True):
        if autograd: self.cal_Jacobian()
        return self.D@strain
    
    def cal_Jacobian(self):
        self.Jacobian=self.D


class OtherMaterial(CommonMaterial):
    """constitutive relationship of other material, you can define it by yourself"""
    pass
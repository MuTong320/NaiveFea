import numpy as np

class LinearMaterial:
    """constitutive relationship of isotropic linear material"""
    def __init__(self,E,nv):
        self.E=E 
        self.nv=nv 
        self.D=self.cal_D()
        self.tangent=self.cal_tangent()

    def cal_D(self):
        E=self.E
        nv=self.nv
        return E/((1.0+nv)*(1.0-2*nv))*np.array([\
            [1.0-nv,nv,0.0], 
            [nv,1.0-nv,0.0], 
            [0.0,0.0,0.5-nv]])
    
    def cal_tangent(self):
        return self.D
    
    def get_stress(self,strain):
        return self.D@strain

class OtherMaterial:
    """constitutive relationship of other material, you can define it by yourself"""
    # recommand using PyTorch or TensorFlow to user for automatically calculate tangent stiffness
    pass
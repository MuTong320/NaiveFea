import numpy as np
import numpy.linalg as LA

from .tools import *


class CommonMaterial2D:
    """
    Common material includes vacuum calculating Jacobian automatically.
    """
    def __call__(self,strain,*variables,autograd=True):
        return self.forward(strain,*variables,autograd)

    def __init__(self):
        self.name='Unnamed material'
        self.tensorform=False
        self.plane_stress=False
        self.use_flatten=False
    
    def set_name(self,name):
        """Set name of the material"""
        self.name=name      

    def forward(self,strain,*variables,autograd=True):
        """
        Calculate stress by given strain and (optional) give the Jacobian of the increment.
        """
        self.strain=strain
        self.variables=variables
        self.stress=self.cal_stress(self.strain,*self.variables)
        if autograd: self.cal_Jacobian()
        return self.stress
    
    def cal_stress(self,strain,*variables):
        """
        Don't update self.stress and use member variables (self.strain and self.variables) in this function.
        """
        if self.tensorform:
            epsilon=self.__cal_strain_tensor(strain)
            sigma=self.cal_sigma(epsilon,*variables)
            return self.__cal_stress_vector(sigma)
        else:
            return np.zeros_like(strain)
    
    def cal_sigma(self,epsilon,*variables):
        """Calculate stress tensor (sigma) by strain tensor (epsilon)."""
        return np.zeros_like(epsilon)

    def __cal_strain_tensor(self,vector):
        return np.array([ 
            [vector[0],0.5*vector[2]],
            [0.5*vector[2],vector[1]]
        ])
    
    def __cal_stress_vector(self,tensor):
        return np.array([tensor[0,0],tensor[1,1],tensor[0,1]])
    
    def cal_Jacobian(self,diff=1e-3):
        self.Jacobian=np.zeros((3,3))
        for component in range(3):
            dstrain=np.zeros(3)
            dstrain[component]=diff
            strain_new = self.strain+dstrain
            stress_new=self.cal_stress(strain_new,*self.variables)
            dstress=stress_new-self.stress
            self.Jacobian[component]=dstress/diff


class LinearElastic(CommonMaterial2D):
    """
    Constitutive relationship of isotropic linear material.
    E: Young's modulus;
    nv: Poisson's ratio.
    """
    def __init__(self,E,nv):
        super().__init__()
        self.E=E 
        self.nv=nv 
        self.cal_D()

    def cal_D(self):
        E=self.E
        nv=self.nv
        if self.plane_stress: self.D=E/(1.0-nv**2)*\
            np.array([
            [1.0,nv,0.0], 
            [nv,1.0,0.0], 
            [0.0,0.0,0.5*(1.0-nv)]
        ])
        else: self.D=E/((1.0+nv)*(1.0-2.0*nv))*\
            np.array([
            [1.0-nv,nv,0.0], 
            [nv,1.0-nv,0.0], 
            [0.0,0.0,0.5*(1.0-2.0*nv)]
        ])
    
    def cal_stress(self,strain):
        return self.D@strain
    
    def cal_Jacobian(self):
        self.Jacobian=self.D


class TensorHookean(CommonMaterial2D):
    """
    mu: shear modulus;
    kappa: volum modulus.
    """
    def __init__(self,mu,kappa):
        super().__init__()
        self.mu=mu
        self.kappa=kappa
        self.tensorform=True
    
    @tensorflatten
    def cal_sigma(self, epsilon, *variables):
        trace=np.trace(epsilon)
        lamda=self.kappa-2.0/3.0*self.mu
        return 2.0*self.mu*epsilon+lamda*trace*np.eye(3)


class OtherMaterial(CommonMaterial2D):
    """
    Constitutive relationship of other material, you can define it by yourself.
    You should difine: 
    1. cal_stress(strain,*variables) or cal_sigma(epsilon,*variables) in tensor form;
    2. (optional) cal_Jacobian(strain,*variables).
    """
    pass
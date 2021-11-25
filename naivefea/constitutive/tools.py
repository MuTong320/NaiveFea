import numpy as np


def flatten(cal_stress3d):
    def cal_stress2d(self,strain2d,*variables):
        strain3d=np.array([strain2d[0],strain2d[1],0.0,strain2d[2],0.0,0.0])
        stress3d=cal_stress3d(strain3d,*variables)
        return np.array([stress3d[0],stress3d[1],stress3d[3]])
    return cal_stress2d

def tensorflatten(cal_sigma3d):
    def cal_sigma2d(self,epsilon2d,*variables):
        epsilon3d=np.array([
            [epsilon2d[0,0],epsilon2d[0,1],0.0],
            [epsilon2d[1,0],epsilon2d[1,1],0.0],
            [0.0,0.0,0.0]
        ])
        sigma3d=cal_sigma3d(self,epsilon3d,*variables)
        return np.array([
            [sigma3d[0,0],sigma3d[0,1]],
            [sigma3d[1,0],sigma3d[1,1]]
        ])
    return cal_sigma2d
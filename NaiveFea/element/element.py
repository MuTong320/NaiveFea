import numpy as np

class Element:
    """plane triangle element"""
    def __init__(self,material,positions):
        """postion=ndarray([[x0,y0],[x1,y1],[x2,y2]])"""
        self.x=positions.T[0]
        self.y=positions.T[1]
        self.node_set=np.array([0,1,2],dtype=np.uint64)
        self.D=material.D
        self.area_2=self.cal_area_2()
        self.B=self.cal_B()
        self.K_element=self.element_integrate()

    def cal_area_2(self):
        x=self.x
        y=self.y
        matrix=np.array([\
            [1.0,x[0],y[0]], 
            [1.0,x[1],y[1]], 
            [1.0,x[2],y[2]]])
        det=np.linalg.det(matrix)
        return abs(det)
    
    def cal_B(self):
        x=self.x
        y=self.y
        return 1/self.area_2*np.array([\
            [y[1]-y[2],0.0,y[2]-y[0],0.0,y[0]-y[1],0.0], 
            [0.0,x[2]-x[1],0.0,x[0]-x[2],0.0,x[1]-x[0]], 
            [x[2]-x[1],y[1]-y[2],x[0]-x[2],y[2]-y[0],x[1]-x[0],y[0]-y[1]]])
    
    def element_integrate(self):
        return self.area_2*self.B.T@self.D@self.B
    
    def set_node_set(self,node_set):
        self.node_set=node_set
    
    def get_strain(self,deform_local):
        return self.B@deform_local

    def get_stress(self,deform_local):
        return self.D@self.B@deform_local
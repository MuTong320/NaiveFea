import numpy as np

class CommonElement:
    """plane triangle element"""
    def __init__(self, positions):
        """postion=ndarray([[x0,y0],[x1,y1],[x2,y2]])"""
        self.x=positions.T[0]
        self.y=positions.T[1]
        self.node_indexes=np.arange(len(positions),dtype=np.uint64)
    
    def set_deform(self,deform_element):
        self.deform_element=deform_element
    
    def set_node_indexes(self,node_indexes):
        self.node_indexes=node_indexes

class CommonTriangleElement(CommonElement):
    """plane triangle element"""
    def __init__(self, positions):
        super().__init__(positions)
        self.double_area=self.__cal_area_2()
        self.B=self.__cal_B()

    def __cal_area_2(self):
        x=self.x
        y=self.y
        matrix=np.array([
            [1.0,x[0],y[0]], 
            [1.0,x[1],y[1]], 
            [1.0,x[2],y[2]]])
        det=np.linalg.det(matrix)
        return abs(det)
    
    def __cal_B(self):
        x=self.x
        y=self.y
        return 1/self.double_area*np.array([
            [y[1]-y[2],0.0,y[2]-y[0],0.0,y[0]-y[1],0.0], 
            [0.0,x[2]-x[1],0.0,x[0]-x[2],0.0,x[1]-x[0]], 
            [x[2]-x[1],y[1]-y[2],x[0]-x[2],y[2]-y[0],x[1]-x[0],y[0]-y[1]]])

class TriangleElement(CommonTriangleElement):
    """plane triangle element for general elastic solid"""
    def __init__(self, material, positions):
        super().__init__(positions)
        self.material=material
    
    def forward(self):
        self.cal_strain()
        self.cal_stress(autograd=True)
        self.cal_Jacobian()
        self.cal_force()
    
    def cal_strain(self):
        self.strain=self.B@self.deform_element

    def cal_stress(self,autograd=False):
        self.material.forward(self.strain,autograd=autograd)
        self.stress=self.material.stress
    
    def cal_Jacobian(self):
        self.element_integrate()
    
    def element_integrate(self):
        self.K_element=0.5*self.double_area*self.B.T@self.material.Jacobian@self.B
    
    def cal_force(self):
        self.force=0.5*self.double_area*self.B.T@self.stress


class SimpleTriangleElement(TriangleElement):
    """plane triangle element for linear elastic solid"""
    def __init__(self, material, positions):
        super().__init__(material, positions)
        self.material=material
        self.element_integrate()
    
    def element_integrate(self):
        self.K_element=0.5*self.double_area*self.B.T@self.material.D@self.B
        
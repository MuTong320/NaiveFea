import numpy as np
from .. import element

class GlobalStiffness:
    """global (tangent) stiffness matrix"""
    def __init__(self,mesh):
        # get mesh data
        self.nodes=mesh.points[:,:2]
        self.elements=mesh.cells_dict['triangle']
        self.meterials=mesh.cell_data
        # initial global variables
        self.len_global=2*len(self.nodes)
        self.deform=np.zeros(self.len_global)
        self.force=np.zeros(self.len_global)
        self.cal_K()
    
    def instant_Element(self,element_index,node_set):
        material=self.meterials[element_index]
        positions=np.array([self.nodes[node_set[j]][:] for j in range(3)])
        return element.Element(material,positions)
    
    def Ke2K(self,element):
        K_element=element.K_element
        node_set=element.node_set
        deform_global_index=np.array([[2*node_set[i],2*node_set[i]+1] \
            for i in range(3)],dtype=np.uint64).reshape(-1)
        for i_local,i_global in enumerate(deform_global_index):
            for j_local,j_global in enumerate(deform_global_index):
                self.K[i_global,j_global]+=K_element[i_local,j_local]
    
    def cal_K(self):
        # initialize K
        self.K=np.zeros((self.len_global,self.len_global))
        # fill K per element
        for element_index,node_set in enumerate(self.elements):
            element=self.instant_Element(element_index,node_set)
            element.set_node_set(node_set)
            self.Ke2K(element)

class ReducedStiffness(GlobalStiffness):
    """reduced (tangent) stiffness matrix"""
    def __init__(self, mesh):
        super().__init__(mesh)
        # conditions
        self.x_fix={}
        self.y_fix={}
        self.f_given={}
    
    def __mark_deform_free(self):
        self.deform_free_index=[]
        for node,_ in enumerate(self.nodes):
            if node not in self.x_fix:
                self.deform_free_index.append(2*node)
            if node not in self.y_fix:
                self.deform_free_index.append(2*node+1)
        self.len_reduce=len(self.deform_free_index)
    
    def __init_global_variables(self):
        for node in self.x_fix:
            self.deform[2*node]=self.x_fix[node]
        for node in self.y_fix:
            self.deform[2*node+1]=self.y_fix[node]
        for node in self.f_given:
            self.force[2*node]=self.f_given[node][0]
            self.force[2*node+1]=self.f_given[node][1]

    def __init_reduce_variables(self):
        self.deform_reduce=np.zeros(self.len_reduce)
        self.force_reduce=np.zeros(self.len_reduce)
        self.K_reduce=np.zeros((self.len_reduce,self.len_reduce))
        for i_reduce,i_global in enumerate(self.deform_free_index):
            self.force_reduce[i_reduce]=self.force[i_global]
        for i_reduce,i_global in enumerate(self.deform_free_index):
            for j_reduce,j_global in enumerate(self.deform_free_index):
                self.K_reduce[i_reduce,j_reduce]=self.K[i_global,j_global]
    
    def reduce_system(self):
        self.__mark_deform_free()
        self.__init_global_variables()
        self.__init_reduce_variables()

    def solve_reduce_system(self):
        self.deform_reduce=np.linalg.solve(self.K_reduce,self.force_reduce)

    def update_global_variables(self):
        for i_reduce,i_global in enumerate(self.deform_free_index):
            self.deform[i_global]=self.deform_reduce[i_reduce]
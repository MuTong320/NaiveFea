import numpy as np
from .. import element


class GlobalSystem:
    """
    The equation system is built for forward.
    """
    def __init__(self,mesh):
        self.simplest=True
        self.material_dict_location=dict()
        self.material_dict=dict()
        self.__get_mesh_data(mesh)

    def __get_mesh_data(self, mesh):
        self.nodes=mesh.points[:,:2]
        self.elements=mesh.cells_dict['triangle']

    # calculation for general material
    def init_global_system(self):
        self.init_global_vars()
        self.init_elements_vars()
        self.init_global_K()

    def init_global_vars(self):
        self.__len_global=2*len(self.nodes)
        self.zero_global_vars()
    
    def zero_global_vars(self):
        self.deform=np.zeros(self.__len_global)
        self.deform_obj=np.zeros(self.__len_global)
        self.force_obj=np.zeros(self.__len_global)
        self.ddeform=np.zeros(self.__len_global)
        self.force=np.zeros(self.__len_global)

    def zero_global_force(self):
        self.force=np.zeros(self.__len_global)

    def init_elements_vars(self):
        self.__len_elements=len(self.elements)
        self.zero_element_vars()

    def zero_element_vars(self):
        self.strain=np.zeros((self.__len_elements,3))
        self.stress=np.zeros((self.__len_elements,3))

    def init_global_K(self):
        self.__len_global=2*len(self.nodes)
        self.zero_global_K()
        if self.simplest: self.__cal_K()

    def zero_global_K(self):
        self.K=np.zeros((self.__len_global,self.__len_global))
    
    def __cal_K(self):
        for element_index,_ in enumerate(self.elements):
            element=self.__instant_Element(element_index)
            self.__Ke2K(element)
    
    def forward(self,element_indexes=None):
        if not bool(element_indexes): element_indexes=range(len(self.elements))
        for element_index in element_indexes:
            element=self.__instant_deformed_Element(element_index)
            self.__element_forward(element_index, element)
            self.__Fe2F(element)
            self.__Ke2K(element)
    
    def __instant_deformed_Element(self, element_index):
        element=self.__instant_Element(element_index)
        node_indexes=self.elements[element_index]
        deform_local=self.__get_deform_list(node_indexes)
        element.set_deform(deform_local)
        return element
    
    def __instant_Element(self,element_index):
        material=self.material_dict_location[element_index]
        node_indexes=self.elements[element_index]
        positions=np.array([self.nodes[node_indexes[j]][:] for j in range(3)])
        if self.simplest:
            element_instance=element.SimpleTriangleElement(material,positions)
        else:
            element_instance=element.TriangleElement(material,positions)
        element_instance.set_node_indexes(node_indexes)
        return element_instance
    
    def __get_deform_list(self,node_indexes):
        deform_local=[[self.deform[2*node],self.deform[2*node+1]] \
            for node in node_indexes.tolist()]
        return np.array(deform_local).reshape(-1)

    def __element_forward(self, element_index, element):
        element.forward()
        self.strain[element_index]=element.strain
        self.stress[element_index]=element.stress

    def __Fe2F(self,element):
        force_element=element.force
        node_indexes=element.node_indexes
        deform_global_index=self.__cal_element_map(node_indexes)
        for i_local,i_global in enumerate(deform_global_index):
            self.force[i_global]+=force_element[i_local]

    def __cal_element_map(self, node_indexes):
        return np.array([[2*node_indexes[i],2*node_indexes[i]+1] \
            for i in range(3)],dtype=np.uint64).reshape(-1)
    
    def __Ke2K(self,element):
        K_element=element.K_element
        node_indexes=element.node_indexes
        deform_global_index=self.__cal_element_map(node_indexes)
        for i_local,i_global in enumerate(deform_global_index):
            for j_local,j_global in enumerate(deform_global_index):
                self.K[i_global,j_global]+=K_element[i_local,j_local]


class ReducedSystem(GlobalSystem):
    """
    The equation system is built for backward.
    """
    def __init__(self, mesh):
        super().__init__(mesh)
    
    # built the reduced strain, stress, and stiffness
    def backward(self):
        self.__init_reduce_system()
        self.__solve_reduce_system()
    
    def __init_reduce_system(self):
        self.__init_reduce_vars()
        self.__update_reduce_force()
        self.__init_reduce_K()

    def __init_reduce_vars(self):
        self.__reduce_ddeform=np.zeros(self.len_reduce)
        self.__reduce_dforce=np.zeros(self.len_reduce)

    def __update_reduce_force(self):
        force=self.force_obj-self.K@self.deform_obj
        if not self.simplest: dforce=force-self.force
        if self.simplest:
            for i_reduce,i_global in enumerate(self.deform_free_index):
                self.__reduce_dforce[i_reduce]=force[i_global]
        else:
            for i_reduce,i_global in enumerate(self.deform_free_index):
                self.__reduce_dforce[i_reduce]=dforce[i_global]

    def __init_reduce_K(self):
        self.__reduce_K=np.zeros((self.len_reduce,self.len_reduce))
        self.__cal_reduce_K()

    def __cal_reduce_K(self):
        for i_reduce,i_global in enumerate(self.deform_free_index):
            for j_reduce,j_global in enumerate(self.deform_free_index):
                self.__reduce_K[i_reduce,j_reduce]=self.K[i_global,j_global]

    def __solve_reduce_system(self):
            self.__reduce_ddeform=np.linalg.solve(self.__reduce_K,self.__reduce_dforce)
            self.error=max(abs(self.__reduce_ddeform))

    def optimize(self):
        self.__update_ddeform()
        self.__update_global_deform()
        self.__update_global_force()

    def __update_ddeform(self):
        for i_reduce,i_global in enumerate(self.deform_free_index):
            self.ddeform[i_global]=self.__reduce_ddeform[i_reduce]

    def __update_global_deform(self):
        self.deform+=self.ddeform

    def __update_global_force(self):
            self.force+=self.K@self.ddeform


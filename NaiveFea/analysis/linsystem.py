import numpy as np
from .. import element


class GlobalSystem:
    """built the global strain, stress, and stiffness"""
    def __init__(self,mesh):
        self.simplest=True
        self.material_dict_location=dict()
        self.material_dict=dict()
        self.__get_mesh_data(mesh)

    def __get_mesh_data(self, mesh):
        self.nodes=mesh.points[:,:2]
        self.elements=mesh.cells_dict['triangle']

    # calculation for general material
    def zero_global_variables(self):
        self.deform=np.zeros(self.__len_global)
        self.force=np.zeros(self.__len_global)
        self.K=np.zeros((self.__len_global,self.__len_global))
        self.strain=np.zeros((self.__len_elements,3))
        self.stress=np.zeros((self.__len_elements,3))
    
    def forward(self,backward=True):
        for element_index,_ in enumerate(self.elements):
            element=self.__new_local_Element(element_index)
            self.__forward_element(element_index, element)
            self.__Fe2F(element)
            if backward: self.__Ke2K(element)

    def __new_local_Element(self, element_index):
        node_set=self.elements[element_index]
        element=self.__instant_Element(element_index)
        deform_local=self.__get_deform_list(node_set)
        element.set_deform(deform_local)
        return element
    
    def __instant_Element(self,element_index):
        material=self.material_dict_location[element_index]
        node_set=self.elements[element_index]
        positions=np.array([self.nodes[node_set[j]][:] for j in range(3)])
        if self.simplest:
            element_instance=element.SimpleTriangleElement(material,positions)
        else:
            element_instance=element.TriangleElement(material,positions)
        element_instance.set_node_set(node_set)
        return element_instance
    
    def __get_deform_list(self,node_set):
        deform_local=[[self.deform[2*node],self.deform[2*node+1]] \
            for node in node_set.tolist()]
        return np.array(deform_local).reshape(-1)

    def __forward_element(self, element_index, element):
        element.forward()
        self.strain[element_index]=element.strain
        self.stress[element_index]=element.stress

    def __Fe2F(self,element):
        force_element=element.B.T@element.stress
        node_set=element.node_set
        deform_global_index=self.__cal_element_map(node_set)
        for i_local,i_global in enumerate(deform_global_index):
            self.force[i_global]+=force_element[i_local]

    def __cal_element_map(self, node_set):
        return np.array([[2*node_set[i],2*node_set[i]+1] \
            for i in range(3)],dtype=np.uint64).reshape(-1)

    # calculation for linear elastic solid
    def init_global_system(self):
        self.__initial_global_variales()
        self.__initial_global_stiffness()

    def __initial_global_variales(self):
        self.__len_global=2*len(self.nodes)
        self.__len_elements=len(self.elements)
        self.zero_global_variables()

    def __initial_global_stiffness(self):
        self.K=np.zeros((self.__len_global,self.__len_global))
        self.__cal_K()
    
    def __cal_K(self):
        for element_index,_ in enumerate(self.elements):
            element=self.__instant_Element(element_index)
            self.__Ke2K(element)
    
    def __Ke2K(self,element):
        K_element=element.K_element
        node_set=element.node_set
        deform_global_index=self.__cal_element_map(node_set)
        for i_local,i_global in enumerate(deform_global_index):
            for j_local,j_global in enumerate(deform_global_index):
                self.K[i_global,j_global]+=K_element[i_local,j_local]


class ReducedSystem(GlobalSystem):
    """reduce global system and its solver"""
    def __init__(self, mesh):
        super().__init__(mesh)
    
    # built the reduced strain, stress, and stiffness
    def __init_reduce_system(self):
        self.__init_reduce_variables()
        self.__update_force_reduce()
        self.__init_reduce_stiffness()

    def __init_reduce_variables(self):
        self.__deform_reduce=np.zeros(self.len_reduce)
        self.__force_reduce=np.zeros(self.len_reduce)
        self.__dforce_reduce=np.zeros(self.len_reduce)

    def __update_force_reduce(self):
        force=self.force-self.K@self.deform
        for i_reduce,i_global in enumerate(self.deform_free_index):
            self.__force_reduce[i_reduce]=force[i_global]

    def __init_reduce_stiffness(self):
        self.__K_reduce=np.zeros((self.len_reduce,self.len_reduce))
        self.__cal_K_reduce()

    def __cal_K_reduce(self):
        for i_reduce,i_global in enumerate(self.deform_free_index):
            for j_reduce,j_global in enumerate(self.deform_free_index):
                self.__K_reduce[i_reduce,j_reduce]=self.K[i_global,j_global]
    
    # solve linear system
    def slove(self):
        self.__init_reduce_system()
        self.__solve_reduce_system()
        self.__update_global_variables()

    def __solve_reduce_system(self):
        self.__deform_reduce=np.linalg.solve(self.__K_reduce,self.__force_reduce)

    def __update_global_variables(self):
        self.__update_global_deform()
        self.__update_global_force()

    def __update_global_deform(self):
        for i_reduce,i_global in enumerate(self.deform_free_index):
            self.deform[i_global]=self.__deform_reduce[i_reduce]

    def __update_global_force(self):
        for i_global in self.deform_free_index:
            self.force[i_global]=self.K[i_global]@self.deform
    
    def __forward_reduce(self):
        pass

    # Newton method
    def newton_iterate(self):
        self.forward()
        self.backward()
        self.optimize()

    def backward(self):
        self.__forward_reduce()
    
    def optimize(self):
        pass

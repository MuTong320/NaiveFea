import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from . import stiffness

class LinearFea(stiffness.ReducedStiffness):
    """the simplest analysis, suits for linear material and only has one increment"""
    def __init__(self, mesh):
        super().__init__(mesh)
        self.init_show_dict()
    
    def init_show_dict(self):
        self.show_dict={'position':{'x':[],'y':[]},\
            'deform':{'Ux':[],'Uy':[]},
            'force':{'Fx':[],'Fy':[]}, 
            'strain':{'e11':[],'e22':[],'e12':[]},
            'stress':{'S11':[],'S22':[],'S12':[]}}
        self.show_dict['position']['x']=self.nodes.T[0]
        self.show_dict['position']['y']=self.nodes.T[1]
    
    def set_deform_conditions(self,x_fix,y_fix):
        self.x_fix.update(x_fix)
        self.y_fix.update(y_fix)

    def set_force_conditions(self,f_given):
        self.f_given.update(f_given)

    def set_conditions(self,x_fix,y_fix,f_given):
        self.set_deform_conditions(x_fix,y_fix)
        self.set_force_conditions(f_given)
    
    def update_show_dict(self):
        self.show_dict['deform']['Ux']=self.deform.reshape(len(self.nodes),2).T[0]
        self.show_dict['deform']['Uy']=self.deform.reshape(len(self.nodes),2).T[1]
        self.show_dict['force']['Fx']=self.force.reshape(len(self.nodes),2).T[0]
        self.show_dict['force']['Fy']=self.force.reshape(len(self.nodes),2).T[1]
        self.post_process()
        for i,name in enumerate(('e11','e22','e12')):
            self.show_dict['strain'][name]=self.strain_list[i]
        for i,name in enumerate(('S11','S22','S12')):
            self.show_dict['stress'][name]=self.stress_list[i]
    
    def get_deform_local(self,node_set):
        deform_local=[]
        for node in node_set:
            u=self.show_dict['deform']['Ux'][node]
            v=self.show_dict['deform']['Uy'][node]
            deform_local.append(u)
            deform_local.append(v)
        return deform_local
    
    def post_process(self):
        self.strain_list=[]
        self.stress_list=[]
        for element_index,node_set in enumerate(self.elements):
            element=self.instant_Element(element_index,node_set)
            deform_local=self.get_deform_local(node_set)
            strain=element.get_strain(deform_local)
            self.strain_list.append(strain)
            stress=element.get_strain(deform_local)
            self.stress_list.append(stress)
        self.strain_list=np.array(self.strain_list).T
        self.stress_list=np.array(self.stress_list).T
    
    def submit(self):
        self.reduce_system()
        self.solve_reduce_system()
        self.update_global_variables()
        self.update_show_dict()
    
    def plot_mesh(self):
        x=self.show_dict['position']['x']
        y=self.show_dict['position']['y']
        mesh_fig=tri.Triangulation(x,y,self.elements)
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.triplot(mesh_fig,'b.-',lw=1)
        plt.title('Mesh')
    
    def plot_color(self,name,component,shading):
        x=self.show_dict['position']['x']
        y=self.show_dict['position']['y']
        z=self.show_dict[name][component]
        fig=tri.Triangulation(x,y,self.elements)
        plt.figure()
        plt.gca().set_aspect('equal')
        plt.tripcolor(fig,z,shading=shading)
        plt.triplot(fig,lw=1)
        plt.colorbar()
        plt.title(f'Plot of FEA result: {name}({component})')
    
    def plot(self,name,component=''):
        if name=='mesh':
            self.plot_mesh()
        if name in ('deform','force'):
            self.plot_color(name,component,'gouraud')
        if name in ('strain','stress'):
            self.plot_color(name,component,'flat')
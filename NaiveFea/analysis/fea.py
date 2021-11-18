import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from naivefea import element

from . import linsystem

class LinearSolver(linsystem.ReducedSystem):
    def __init__(self, mesh):
        super().__init__(mesh)


class LinearFeaCommand(LinearSolver):
    """
    the simplest analysis, suits for linear material and only has one increment. 
    without plot function.
    """
    def __init__(self, mesh):
        super().__init__(mesh)
        self.solved=False
        self.__init_condition()
        self.__init_show_dict()

    def __init_condition(self):
        self.x_given=set()
        self.y_given=set()
        self.x_given_displace=dict()
        self.y_given_displace=dict()
        self.f_given=dict()
    
    def __init_show_dict(self):
        self.__init_reference_dict()
        self.__init_deformed_dict()

    def __init_reference_dict(self):
        self.reference_dict={'position':{'x':[],'y':[]},'material':{'name':[],'color':[]}}
        self.reference_dict['position']['x']=self.nodes.T[0]
        self.reference_dict['position']['y']=self.nodes.T[1]

    def __init_deformed_dict(self):
        self.current_dict={'position':{'x':[],'y':[]},
            'deform':{'Ux':[],'Uy':[]},
            'force':{'Fx':[],'Fy':[]}, 
            'strain':{'e11':[],'e22':[],'e12':[]},
            'stress':{'S11':[],'S22':[],'S12':[]}}
    
    # preprocess
    #   set material
    def uniform_material(self,material,element_set='all'):
        """assign the given material to all elements of current analysis."""
        self.check_solved()
        if type(element_set)==str and element_set=='all':
            for element_index,_ in enumerate(self.elements):
                self.__update_material(material, element_index)
        else:
            for element_index in element_set:
                self.__update_material(material, element_index)

    def __update_material(self, material, element_index):
        if element_index > len(self.elements): raise ValueError
        if material not in self.materials_location.values() \
            and material.name in self.materials.values():
            material.name=material.name+'*'
        self.materials_location.update({element_index:material})
        self.materials.update({element_index:material.name})
    
    #   set boundary condition
    def set_deform_conditions(self,operation='fix',Ux=set(),Uy=set(),Uxy=set()):
        """assign deformation on the given nodes."""
        self.check_solved()
        if operation=='fix': 
            self.__fix_node_deform(Ux,Uy)
            self.__fix_node_deform(Uxy,Uxy)
        elif operation=='displace': 
            if type(Ux)==dict: self.__displace_for_x(Ux)
            if type(Uy)==dict: self.__displace_for_y(Uy)
            if type(Uxy)==dict: self.__displace_for_xy(Uxy)
        else:
            raise ValueError

    def __fix_node_deform(self, x_fix, y_fix):
        self.x_given.update(x_fix)
        self.y_given.update(y_fix)
        for x in x_fix:
            self.x_given_displace.update({x:0.0})
        for y in y_fix:
            self.y_given_displace.update({y:0.0})
    
    def __displace_for_x(self,x_displace):
        self.x_given_displace.update(x_displace)
        self.x_given.update(x_displace.keys())
    
    def __displace_for_y(self,y_displace):
        self.y_given_displace.update(y_displace)
        self.y_given.update(y_displace.keys())

    def __displace_for_xy(self,xy_condition):
        x_dict=dict()
        y_dict=dict()
        for node,displace in xy_condition.items():
            x_dict.update({node:displace[0]})
            y_dict.update({node:displace[1]})
        self.__displace_for_x(x_dict)
        self.__displace_for_y(y_dict)

    def set_force_conditions(self,f_given):
        """assign force load on the given nodes."""
        self.check_solved()
        self.f_given.update(f_given)
    
    def set_equation(self):
        pass
    
    def clear_conditions(self,*name):
        if 'Ux' in name:
            self.x_given=set()
            self.x_given_displace=dict()
        if 'Uy' in name:
            self.y_given=set()
            self.y_given_displace=dict()
        if 'Uxy' in name:
            self.x_given=set()
            self.x_given_displace=dict()
            self.y_given=set()
            self.y_given_displace=dict()
        if 'F' in name:
            self.f_given=dict()
        if 'all' in name:
            self.x_given=set()
            self.x_given_displace=dict()
            self.y_given=set()
            self.y_given_displace=dict()
            self.f_given=dict()
    
    def clear_node_conditions(self,node,*name):
        if 'Ux' in name:
            self.x_given.discard(node)
            self.x_given_displace.pop(node,0)
        if 'Uy' in name:
            self.y_given.discard(node)
            self.y_given_displace.pop(node,0)
        if 'Uxy' in name:
            self.x_given.discard(node)
            self.x_given_displace.pop(node,0)
            self.y_given.discard(node)
            self.y_given_displace.pop(node,0)
        if 'F' in name:
            self.f_given.discard(node)
        if 'all' in name:
            self.x_given.discard(node)
            self.x_given_displace.pop(node,0)
            self.y_given.discard(node)
            self.y_given_displace.pop(node,0)
            self.f_given.pop(node,0)

    def check_solved(self):
        if self.solved: 
            print('Conditions may have been changed! Please resubmit for new result.')
            self.sloved=False
    
    #   submit preprocess
    def preprocess(self):
        self.init_global_system()
        self.__init_reduce_map()
        self.__fix_global_variables()

    def __init_reduce_map(self):
        self.deform_free_index=[]
        self.__cal_reduce_map()
        self.len_reduce=len(self.deform_free_index)
    
    def __cal_reduce_map(self):
        for node,_ in enumerate(self.nodes):
            if node not in self.x_given:
                self.deform_free_index.append(2*node)
            if node not in self.y_given:
                self.deform_free_index.append(2*node+1)
    
    def __fix_global_variables(self):
        for node,displace in self.x_given_displace.items():
            self.deform[2*node]=displace
        for node,displace in self.y_given_displace.items():
            self.deform[2*node+1]=displace
        for node,force in self.f_given.items():
            self.force[2*node],self.force[2*node+1]=force
    
    # solving
    def submit(self,postprocess=True,nonlinear=False):
        """Before submit, set the boundary condition first."""
        self.preprocess()
        if nonlinear:
            self.nonlinear_solver()
        else:
            self.linear_solver()
        if postprocess:
            self.forward()
            self.update_show_dict()

    def linear_solver(self):
        self.slove()
        self.solved=True

    def nonlinear_solver(self):
        self.simplest=False
        self.init_forward()
    
    def init_forward(self):
        self.zero_global_variables()
        self.forward()
        self.slove()
        self.solved=True
    
    # postprecess
    
    #   data dict for viewing and ploting
    def update_show_dict(self):
        self.__update_show_deform()
        self.__update_show_position()
        self.__update_show_force()
        self.__update_show_elements_variable()

    def __update_show_deform(self):
        self.current_dict['deform']['Ux']=self.deform.reshape(len(self.nodes),2).T[0]
        self.current_dict['deform']['Uy']=self.deform.reshape(len(self.nodes),2).T[1]
    
    def __update_show_position(self):
        self.current_dict['position']['x']=self.reference_dict['position']['x']\
            +self.current_dict['deform']['Ux']
        self.current_dict['position']['y']=self.reference_dict['position']['y']\
            +self.current_dict['deform']['Uy']

    def __update_show_force(self):
        self.current_dict['force']['Fx']=self.force.reshape(len(self.nodes),2).T[0]
        self.current_dict['force']['Fy']=self.force.reshape(len(self.nodes),2).T[1]

    def __update_show_elements_variable(self):
        strain_list=np.array(self.strain).T
        stress_list=np.array(self.stress).T
        for i,name in enumerate(('e11','e22','e12')):
            self.current_dict['strain'][name]=strain_list[i]
        for i,name in enumerate(('S11','S22','S12')):
            self.current_dict['stress'][name]=stress_list[i]
    
    def get_data(self,name,index):
        """index can be node index or relement index."""
        dict=self.current_dict[name]
        return np.array([dict[key][index] for key in dict])
    
    def cal_variable(self,name):
        """calculate useful variables, including Mises stress, ..."""
        if name=='Mises':
            self.__cal_Mises()
        elif name=='Tresca':
            self.__cal_Tresca()
        else:
            raise ValueError

    def __cal_Mises(self):
        s11=self.current_dict['stress']['S11']
        s22=self.current_dict['stress']['S22']
        s12=self.current_dict['stress']['S12']
        Mises=np.sqrt(0.5*(s11**2+s22**2+(s11-s22)**2+6*s12**2))
        self.current_dict['stress']['Mises']=Mises

    def __cal_Tresca(self):
        s11=self.current_dict['stress']['S11']
        s22=self.current_dict['stress']['S22']
        s12=self.current_dict['stress']['S12']
        Tresca=np.sqrt((s11-s22)**2+4*s12**2)
        self.current_dict['stress']['Tresca']=Tresca


class LinearFea(LinearFeaCommand):
    """
    the simplest analysis, suits for linear material and only has one increment. 
    firstly, set the boundary condition;
    secondly, submit the task;
    finally, view or plot the result.
    """
    def __init__(self, mesh):
        super().__init__(mesh)
        self.set_figsize()

    def set_figsize(self,figsize='small'):
        """figure size can be 'small', 'medium', 'large', 'verylarge', or (length,width)."""
        if figsize=='small':
            self.figsize=(4,4)
        elif figsize=='medium':
            self.figsize=(8,8)
        elif figsize=='large':
            self.figsize=(12,12)
        elif figsize=='verylarge':
            self.figsize=(16,16)
        else:
            self.figsize=figsize
    
    #   plot result figures
    def plot_mesh(self,node=True,element=False,deformed=False,magnification='auto'):
        if not deformed:
            self.__plot_undeformed_mesh(node,element)
        else:
            self.__plot_deformed_mesh(node,element,magnification)

    def __plot_undeformed_mesh(self,node=True,element=False):
        x=self.reference_dict['position']['x']
        y=self.reference_dict['position']['y']
        mesh_fig=tri.Triangulation(x,y,self.elements)
        plt.figure(figsize=self.figsize)
        plt.gca().set_aspect('equal')
        plt.triplot(mesh_fig,'k.-',lw=1)
        if node: self.__plot_node_index(x,y)
        if element: self.__plot_element_index(x,y)
        plt.title('Undeformed Mesh')
    
    def __plot_deformed_mesh(self,node=False,element=False,magnification='auto'):
        x,y,magnification=self.__amplify_deform(magnification)
        mesh_fig=tri.Triangulation(x,y,self.elements)
        plt.figure(figsize=self.figsize)
        plt.gca().set_aspect('equal')
        plt.triplot(mesh_fig,'k.-',lw=1)
        if node: self.__plot_node_index(x,y)
        if element: self.__plot_element_index(x,y)
        plt.title(f'Deformed Mesh \n(magnification = {magnification:.2e})')

    def __amplify_deform(self,magnification):
        if magnification=='auto': magnification=self.__cal_magnification()
        x=self.reference_dict['position']['x']+magnification*self.current_dict['deform']['Ux']
        y=self.reference_dict['position']['y']+magnification*self.current_dict['deform']['Uy']
        return x,y,magnification
    
    def __cal_magnification(self):
        max_position=max(max(abs(self.reference_dict['position']['x'])),\
            max(abs(self.reference_dict['position']['y'])))
        max_deform=max(max(abs(self.current_dict['deform']['Ux'])),\
            max(abs(self.current_dict['deform']['Uy'])))
        return 0.1*max_position/max_deform
    
    def __plot_node_index(self,x,y):
        for index,_ in enumerate(self.nodes):
            plt.annotate(index,(x[index],y[index]),color="red")
    
    def __plot_element_index(self,x,y):
        for index,element in enumerate(self.elements):
            x_mean=1/3*(x[element[0]]+x[element[1]]+x[element[2]])
            y_mean=1/3*(y[element[0]]+y[element[1]]+y[element[2]])
            plt.annotate(index,(x_mean,y_mean),xytext=(x_mean,y_mean))
    
    def plot_restrict(self,fix=True,load=True,node=False,element=False):
        x=self.reference_dict['position']['x']
        y=self.reference_dict['position']['y']
        mesh_fig=tri.Triangulation(x,y,self.elements)
        plt.figure(figsize=self.figsize)
        plt.gca().set_aspect('equal')
        plt.triplot(mesh_fig,'k.-',lw=1)
        if fix:
            self.__plot_x_fix()
            self.__plot_y_fix()
        if load:
            self.__plot_f_given()
        if node: self.__plot_node_index(x,y)
        if element: self.__plot_element_index(x,y)
        plt.title('Load and Restrict')
    
    def __plot_x_fix(self):
        x_fix_position=[]
        for node in self.x_given:
            x_fix_position.append(self.nodes[node])
        x_fix_position=np.array(x_fix_position).T
        x=x_fix_position[0]
        y=x_fix_position[1]
        plt.scatter(x,y,s=100,c='b',marker='>')
    
    def __plot_y_fix(self):
        y_fix_position=[]
        for node in self.y_given:
            y_fix_position.append(self.nodes[node])
        y_fix_position=np.array(y_fix_position).T
        x=y_fix_position[0]
        y=y_fix_position[1]
        plt.scatter(x,y,s=100,c='r',marker='^')
    
    def __plot_f_given(self):
        magnification=self.__cal_force_arrow()
        for node in self.f_given:
            x,y=self.nodes[node]
            dx=magnification*self.f_given[node][0]
            dy=magnification*self.f_given[node][1]
            head_width=0.3*max(abs(dx),abs(dy))
            plt.arrow(x,y,dx,dy,head_width=head_width,fc='g',ec='g')
    
    def __cal_force_arrow(self):
        max_position=max(max(abs(self.reference_dict['position']['x'])),\
            max(abs(self.reference_dict['position']['y'])))
        max_force=max([max(map(abs,value)) for value in self.f_given.values()])
        return 0.1*max_position/max_force
    
    def plot_material(self,node=False,element=False,deformed=False,magnification='auto'):
        self.__init_show_material()
        self.__plot_color('material','color','flat',deformed,magnification)

    def __init_show_material(self):
        self.reference_dict['material']={'name':[],'color':[]}
        material_color={} #name:color
        for name in self.materials.values():
            self.reference_dict['material']['name'].append(name)
            if name not in material_color:
                material_color.update({name:len(material_color)})
            color=material_color[name]
            self.reference_dict['material']['color'].append(color)
    
    def plot(self,name,component='',node=False,element=False,deformed=True,magnification='auto'):
        if name=='mesh':
            self.plot_mesh(node,element,deformed,magnification)
        elif name in ('deform','force'):
            self.__plot_color(name,component,'gouraud',deformed,magnification)
        elif name in ('strain','stress'):
            self.__plot_color(name,component,'flat',deformed,magnification)
        else:
            show_data=self.current_dict[name][component]
            if len(show_data)==len(self.nodes):
                self.__plot_color(name,component,'gouraud',deformed,magnification)
            if len(show_data)==len(self.elements):
                self.__plot_color(name,component,'flat',deformed,magnification)
    
    def __plot_color(self,name,component,shading,deformed,magnification):
        if not deformed:
            x=self.reference_dict['position']['x']
            y=self.reference_dict['position']['y']
        else:
            x,y,magnification=self.__amplify_deform(magnification)
        if name in self.current_dict.keys():
            z=self.current_dict[name][component]
        if name in self.reference_dict.keys():
            z=self.reference_dict[name][component]
        fig=tri.Triangulation(x,y,self.elements)
        plt.figure(figsize=self.figsize)
        plt.gca().set_aspect('equal')
        plt.tripcolor(fig,z,shading=shading)
        plt.triplot(fig,lw=1)
        plt.colorbar()
        title=f'{name.capitalize()} ({component})'
        if deformed:  title+=f'\n(magnification = {magnification:.2e})'
        plt.title(title)
    
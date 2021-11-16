import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from . import nonlinsystem

class AnalysisStep(nonlinsystem.ReducedSystem):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.tolerance=1.0e-4
        self.max_cycle_number=10
        self.steps=1
        self.strain_old=np.zeros(6)
        self.strain_new=np.zeros(6)
        self.stress_old=np.zeros(6)
        self.stress_new=np.zeros(6)

    def set_conditions(self,conditions):
        # condistions={time0:{'x_fix':{...},'y_fix':{...},'f_given':{...}},...}
        self.conditions=conditions

    def get_conditions(self,time):
        self.x_fix=self.conditions[time]['x_fix']
        self.y_fix=self.conditions[time]['y_fix']
        self.f_given=self.conditions[time]['f_given']

    def submit(self):
        for time in self.conditions:
            self.get_conditions(time)
            self.reduce_system()
            self.solve_reduce_system()
            self.update_global_variables()

    def newton_iterate(self):
        for _ in range(self.max_cycle_number):
            pass
            #if calculation_error<self.tolerance: break
            
    

class IncrementalStep(AnalysisStep):
    def __init__(self, mesh):
        super().__init__(mesh)
    


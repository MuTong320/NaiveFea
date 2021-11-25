from .processor import PreProcessor,PostProcessor,PlotProcessor
from .solver import LinearSolver,OneStepSolver


class LinearFea(LinearSolver,PreProcessor,PostProcessor,PlotProcessor):
    """
    The simplest analysis, suits for linear material and only has one increment. 
    Firstly, set the boundary condition;
    Secondly, submit the task;
    Finally, view or plot the result.
    """
    def __init__(self, mesh):
        super().__init__(mesh)
    
    def submit(self,postprocess=True,nonlinear=False):
        """Before submit, set the boundary condition first."""
        self.preprocess()
        self.slove()
        if postprocess:
            self.forward()
            self.update_show_dict()


class OneStepFea(OneStepSolver,PreProcessor,PostProcessor,PlotProcessor):
    """
    The simplest analysis, suits for elastic material and only has one increment. 
    Firstly, set the boundary condition;
    Secondly, submit the task;
    Finally, view or plot the result.
    """
    def __init__(self, mesh):
        super().__init__(mesh)
    
    def submit(self,postprocess=True,nonlinear=False):
        """Before submit, set the boundary condition first."""
        self.preprocess()
        self.slove()
        if not self.solved: print('Unsuccessful!')
        if postprocess:
            self.forward()
            self.update_show_dict()
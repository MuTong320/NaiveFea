from multiprocessing import Pool, Process, process
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import math

from . import kernel


class ParallelSupport(kernel.GlobalSystem):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.processor_number=1
    
    def set_parallel(self,processor_number=None):
        if bool(processor_number): processor_number=multiprocessing.cpu_count()
        self.processor_number=processor_number

    def para_global_K(self):
        self.shm_K=SharedMemory(create=True,size=self.K.nbytes)
        self.K=np.ndarray(self.K.shape,self.K.dtype,self.shm_K.buf)
        self.K[:]=0.0
    
    def para_global_force(self):
        self.shm_force=SharedMemory(create=True,size=self.force.nbytes)
        self.force=np.ndarray(self.force.shape,self.force.dtype,self.shm_force.buf)
        self.force[:]=0.0
    
    def cut_elements(self):
        batch_size=math.ceil(len(self.elements)//self.processor_number)
        self.__batch_list=[batch*batch_size for batch in range(self.processor_number)]
        self.__batch_list.append(len(self.elements))
        start=self.__batch_list[:-1]
        end=self.__batch_list[1:]
        return map(range,start,end)
    
    def para_forward(self):
        self.para_global_force()
        self.para_global_K()
        element_indexes_list=self.cut_elements()
        with Pool(self.processor_number) as pool:
            pool.map(self.batch_forward,element_indexes_list)
        self.unlink_share_memory()

    def unlink_share_memory(self):
        self.K=self.K.copy()
        self.force=self.force.copy()
        self.shm_K.close()
        self.shm_force.close()
        self.shm_K.unlink()
        self.shm_force.unlink()

    def batch_forward(self,element_indexes):
        shm_K=SharedMemory(self.shm_K.name)
        self.K=np.ndarray(self.K.shape,self.K.dtype,shm_K.buf)
        shm_force=SharedMemory(self.shm_force.name)
        self.force=np.ndarray(self.force.shape,self.force.dtype,shm_force.buf)
        super().forward(element_indexes)

    def forward(self,element_indexes=None):
        if self.processor_number==1 or bool(element_indexes)!=None:
            super().forward(element_indexes)
        else:
            self.para_forward()


class LinearSolver(kernel.ReducedSystem,ParallelSupport):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.solved=False

    def slove(self):
        self.backward()
        self.optimize()
        self.solved=True


class OneStepSolver(kernel.ReducedSystem,ParallelSupport):
    def __init__(self, mesh):
        super().__init__(mesh)
        self.simplest=False
        self.solved=False
        self.max_cycles=10
        self.tolerent_error=1e-8

    def slove(self):
        for _ in range(self.max_cycles):
            self.zero_global_K()
            self.zero_global_force()
            self.forward()
            self.backward()
            self.optimize()
            if self.error<self.tolerent_error: 
                self.solved=True
                break
from ..constitutive import *

from .steel import *


def choose(name,grade='default',type='linear'):
    dict=globals()[name]
    if type=='linear':
        E=dict[grade]['E']
        nv=dict[grade]['nv']
        material=LinearElastic(E,nv)
        material.name=f'{grade} {name}'
        return material
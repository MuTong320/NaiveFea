# NaiveFea

A simple python library for 2D finite element analysis.

You can learn how to use it by example.ipynb and example_nonlinear.ipynb. 

* Linear and nonlinear elasticity;
* Easy to calculate and plot user-defined showing data;
* Support multiprocessor;
* Plot mesh, undeformed, and deformed figure, where magnificient of deformed figure can be calculate automatically.

## Pre-processing

Plot figure of mesh and restirct condition

![mesh plot](https://github.com/MuTong320/NaiveFea/blob/master/figure/fig_mesh.png)

![restrict plot](https://github.com/MuTong320/NaiveFea/blob/master/figure/fig_restrict.png)

## Solving

Solving a FEA problem within 8 lines.

```python
mesh=meshio.read('abaqus_mesh.inp')
fea=LinearFea(mesh)
material=LinearElastic(E=10.0,nv=0.3)
fea.uniform_material(material)
fea.set_deform_conditions('fix',Uxy=[0,5,10,15,20])
fea.set_force_conditions({14:(0.001,0)})
fea.submit()
fea.plot('stress','S12')
```

## Post-processing

Easy to plot figures.

![deform mesh plot](https://github.com/MuTong320/NaiveFea/blob/master/figure/fig_deform_mesh.png)

![Ux plot](https://github.com/MuTong320/NaiveFea/blob/master/figure/fig_Ux.png)     ![S12 plot](https://github.com/MuTong320/NaiveFea/blob/master/figure/fig_S12.png)

## Result of two material fea

Ability to solve multi material problem.

![e11 plot](https://github.com/MuTong320/NaiveFea/blob/master/figure/figure_e11.png)

## 



****

## Mark

This project is a teaching material for students who want to implement FEA with Python in [a post on Chaoli forum](https://chaoli.club/index.php/6884) (in Chinese):

[Theory document of linear finite element analysis](https://github.com/MuTong320/NaiveFeaDocument) (in Chinese)

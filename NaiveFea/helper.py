def uniform_material(mesh,material):
        for i,_ in enumerate(mesh.cells_dict['triangle']):
            mesh.cell_data.update({i:material})
from sfepy import data_dir
import numpy as np
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.discrete import (FieldVariable, Material, Integral, Function, Equation, Equations, Problem)
from sfepy.discrete.fem import Mesh, FEDomain, Field
from sfepy.terms import Term
from sfepy.discrete.conditions import Conditions, EssentialBC
from sfepy.solvers.ls import ScipyDirect
from sfepy.solvers.nls import Newton
from sfepy.base.base import IndexedStruct

# Load the mesh file
mesh = Mesh.from_file('hyprb.mesh')
# mesh = Mesh.from_file(data_dir + '/meshes/3d/cube_medium_hexa.mesh')
output_dir = '/mnt/c/Users/jcsta/Documents/BYU/Win. 2023/ME EN 575/Python/Project/vtk'
domain = FEDomain('domain', mesh)
min_z, max_z = domain.get_mesh_bounding_box()[:, 2]
eps = 1e-8 * (max_z - min_z)
omega = domain.create_region('Omega', 'all')
gamma1 = domain.create_region('Gamma1', 'vertices in z < %.10f' % (min_z + eps), 'facet')
gamma2 = domain.create_region('Gamma2', 'vertices in z > %.10f' % (max_z - eps), 'facet')

# options = Options('output_dir', output_dir)

# Define the field variable
field = Field.from_args('fu', np.float64, 'vector', omega, approx_order=3)
u = FieldVariable('u', 'unknown', field)
v = FieldVariable('v', 'test', field, primary_var_name='u')

# Define the material
material = Material('m', D=stiffness_from_youngpoisson(3, 1.0, 0.3))

# Define the integral
integral = Integral('i', order=3)

# Define the term
t1 = Term.new('dw_lin_elastic(m.D, v, u)', integral, omega, m=material, v=v, u=u)
# t2 = Term.new('dw_volume_lvf(m.f, v)', integral, omega, m=material, v=v)
eq = Equation('balance', t1)
eqs = Equations([eq])

# Define the boundary conditions
fix_u = EssentialBC('fix_u', gamma1, {'u.all': 0.0})
compress_u = EssentialBC('compress_u', gamma2, {'u.0': -100})
bc = Conditions([fix_u, compress_u])

# Define the solver
ls = ScipyDirect({})
nls_status = IndexedStruct()
nls = Newton({}, lin_solver=ls, status=nls_status)

# Create the problem
pb = Problem('elasticity', equations=eqs)
pb.save_regions_as_groups('regions')
pb.set_bcs(ebcs=bc)
pb.set_solver(nls)
status = IndexedStruct()
variables = pb.solve(status=status)
print('Nonlinear solver status:\n', nls_status)
print('Stationary solver status:\n', status)
pb.save_state('compression.vtk', variables)

# options.update({'post_process_hook' : 'stress_strain',})

# Solve the problem
# status = pb.solve()
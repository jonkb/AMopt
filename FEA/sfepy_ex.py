r"""
Compressively loaded 3-D cube.

Find :math:`\ul{u}` such that:

.. math::
    \int_{\Omega} D_{ijkl}\ e_{ij}(\ul{v}) e_{kl}(\ul{u})
    = 0
    \;, \quad \forall \ul{v} \;,

where

.. math::
    D_{ijkl} = \mu (\delta_{ik} \delta_{jl}+\delta_{il} \delta_{jk}) +
    \lambda \ \delta_{ij} \delta_{kl}
    \;.
"""
from __future__ import absolute_import
from sfepy.mechanics.matcoefs import stiffness_from_youngpoisson
from sfepy.mechanics.matcoefs import lame_from_youngpoisson
from sfepy.discrete.fem.utils import refine_mesh
from sfepy import data_dir

# Fix the mesh file name if you run this file outside the SfePy directory.
# filename_mesh = 'hyprb.mesh'
filename_mesh = data_dir + '/meshes/3d/cube_medium_hexa.mesh'
refinement_level = 0
filename_mesh = refine_mesh(filename_mesh, refinement_level)

output_dir = '/mnt/c/Users/jcsta/Documents/BYU/Win. 2023/ME EN 575/Python/Project' # set this to a valid directory you have write access to

young = 3.5 # Young's modulus [GPa]
poisson = 0.3  # Poisson's ratio

options = {
    'output_dir' : output_dir,
}

regions = {
    'Omega' : 'all',

    'Left': ('vertices in x < -0.001', 'facet'),
    'Right': ('vertices in x > 0.001', 'facet'),

    'Back': ('vertices in y < -0.001', 'facet'),
    'Front': ('vertices in y > 0.001', 'facet'),

    'Bottom': ('vertices in z < -0.001', 'facet'),
    'Top': ('vertices in z > 0.001', 'facet'),
}

dim = 3

materials = {
    'PLA' : ({'D': stiffness_from_youngpoisson(dim, young, poisson)},),
    'Load' : ({'.val' : [0.0, 0.0, -1000.0]},),
}

# materials = {
#     'PLA' : ({
#         'lam' : lame_from_youngpoisson(dim, young, poisson)[0],
#         'mu' : lame_from_youngpoisson(dim, young, poisson)[1],
#     },),
#     'Load' : ({'.val' : [0.0, 0.0, -1000.0]},),
# }

fields = {
    'displacement': ('real', 'vector', 'Omega', 1),
}

equations = {
   'balance_of_forces' :
   """dw_lin_elastic.2.Omega(PLA.D, v, u)
      = dw_point_load.0.Top(Load.val, v)""",
}

variables = {
    'u' : ('unknown field', 'displacement', 0),
    'v' : ('test field', 'displacement', 'u'),
}

ebcs = {
    'ZSym' : ('Bottom', {'u.2' : 0.0}),
}

solvers = {
    'ls' : ('ls.scipy_direct', {}),
    'newton' : ('nls.newton', {
        'i_max' : 1,
        'eps_a' : 1e-6,
    }),
}

def stress_strain(out, pb, state, extend=False):
    """
    Calculate and output strain and stress for given displacements.
    """
    from sfepy.base.base import Struct

    ev = pb.evaluate
    strain = ev('ev_cauchy_strain.2.Omega(u)', mode='el_avg')
    stress = ev('ev_cauchy_stress.2.Omega(PLA.D, u)', mode='el_avg',
                copy_materials=False)

    out['cauchy_strain'] = Struct(name='output_data', mode='cell',
                                  data=strain, dofs=None)
    out['cauchy_stress'] = Struct(name='output_data', mode='cell',
                                  data=stress, dofs=None)

    return out

PLA = materials['PLA'][0]
PLA.update({'D' : stiffness_from_youngpoisson(3, young, poisson)})
options.update({'post_process_hook' : 'stress_strain',})
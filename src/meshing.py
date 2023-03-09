import subprocess
import numpy as np
import sfepy.mesh.mesh_generators as femg # Used for creating voxel mesh
from skimage import measure as skim
from stl.mesh import Mesh as npstlMesh
import gmsh # Used in stl2msh_py
from util import vprnt, tic, toc

def x2mesh(x, tag, dim="cube", out_format="mesh"):
    """ Convert the given input vector x into a mesh file.
    
    1. Unflatten x into a 3d rho matrix
    2. Threshold rho to create an isosurface, surf: surf = rho2isosurf(rho)
    3. Convert surf to an stl: isosurf2stl(surf)
    4. Convert that stl into a 3D volume & into a mesh with gmsh: stl2msh
    5. Convert that gmsh mesh (msh file) to a SfePy mesh file
        NOTE / TODO: Could also convert to vtk
    
    The following files will be created:
        f"{tag}.stl" (Intermediate)
        f"{tag}.msh" (Intermediate)
        [if out_format == "mesh"] f"{tag}.mesh" (Final Output)
        [if out_format == "vtk"] f"{tag}.vtk" (Final Output)
    
    Parameters
    ----------
        x (np array of float): density function, flattened
        dim ("cube") or (tuple of int, length 3): dimensions of the volume
        tag (str): filename tag
        out_format (str, one of ["msh", "mesh", "vtk"]): Output file format
    """
    
    # Check input
    if not (out_format in ("msh", "mesh", "vtk")):
        raise Exception("Invalid mesh format out_format")

    # Start timing
    times = tic()

    # Unflatten x
    N = x.size
    if dim == "cube":
        nx = np.cbrt(N)
        assert nx % 1 == 0, ("For dim='cube', the length of x must be a cubic"
            f" number. N = {N}")
        nx = int(nx)
        rho = x.reshape((nx, nx, nx))
    else:
        rho = x.reshape(dim)
    
    # Convert rho to an isosurf
    vprnt("Convert density function to isosurface")
    surf = rho2isosurf(rho)
    toc(times, "Converting to isosurface")

    # Convert rho to an stl
    fn_stl = f"{tag}.stl"
    vprnt("Converting isosurface to stl")
    isosurf2stl(surf, fn_stl)
    vprnt(f"Saved to {fn_stl}")
    toc(times, "Converting & saving to stl")

    # Convert stl to gmsh mesh
    fn_msh = f"{tag}.msh"
    vprnt("Convert stl to volume and mesh")
    stl2msh(fn_stl, fn_msh)
    vprnt(f"Saved to {fn_msh}")
    toc(times, "Meshing")

    # Convert to .mesh or .vtk
    if out_format in ("mesh", "vtk"):
        fn_sfepy = f"{tag}.{out_format}"
        vprnt(f"Convert .msh to .{out_format}")
        msh2sfepy(fn_msh, fn_sfepy)
        vprnt(f"Saved to {fn_sfepy}")
        toc(times, f"Converting to .{out_format}")

    # Report total time
    toc(times, f"Total x->{out_format}", total=True)

def rho2isosurf(rho, rho_cutoff=0.5, spacing=(2.,2.,2.), face_thickness=1):
    """ Take a discrete density function rho and convert it to an isosurface
        by thresholding at rho_cutoff. Also adds the top and bottom faces of 
        the cube.

    Uses the marching cubes algorithm from Scikit-image.
    
    Parameters
    ----------
    rho (np.array, shape nx x ny x nz): Density function sampled in a 3d grid.
        This array should not include the top and bottom faces
    rho_cutoff (float): Cutoff threshold for determining whether the density 
        is high enough that we should place material there.
    spacing (iterable, length 3): spacing = (dx, dy, dz)
        This determines the scaling from grid indices to physical units.
    face_thickness (int): Thickness of the top and bottom faces to be added,
        in number of voxels. Minimum is 1. This limitation is because the 
        marching cubes algorithm doesn't support variable element spacing.

    Returns
    -------
    isosurf = [verts, faces, normals, values]
        See https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes
        for a description of those 4 arrays.
    """

    ## Modify density function to include faces & empty "walls"
    rho_shape = np.array(rho.shape)
    # Add space for cube walls & top & bottom
    rho_shape[0] += 2
    rho_shape[1] += 2
    rho_shape[2] += 2*(1+face_thickness) # Z-direction has faces added on
    rho_full = np.zeros(rho_shape)
    rho_full[1:-1, 1:-1, (1+face_thickness):-(1+face_thickness)] = rho
    # Set top & bottom faces to 1
    rho_full[1:-1,1:-1,1:(face_thickness+1)] = 1
    rho_full[1:-1,1:-1,(-1-face_thickness):-1] = 1
    
    ## Run marching cubes
    verts, faces, normals, values = skim.marching_cubes(rho_full, rho_cutoff, 
        spacing=spacing)
    # NOTE: The units for position here are all integer indexes... I think.
    #   May need to convert back -- TODO
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#marching-cubes
    # There's an option for that (spacing). Also play with the other options.
    #   Could try allow_degenerate = False, but it'd probably slow it down
    #       with no benefits to us, unless GMSH is choking
    #   method: {‘lewiner’, ‘lorensen’} -- IDK which is better / faster
    isosurf = [verts, faces, normals, values]
    return isosurf

def isosurf2stl(isosurf, stl_fname):
    """ Convert an isosurface to an STL with numpy-stl
    isosurf = [verts, faces, normals, values]
    """
    # Unpack isosurf
    verts, faces, normals, values = isosurf
    # Use numpy-stl
    # https://numpy-stl.readthedocs.io/en/latest/usage.html#creating-mesh-objects-from-a-list-of-vertices-and-faces
    cm = npstlMesh(np.zeros(faces.shape[0], dtype=npstlMesh.dtype))
    cm.vectors = verts[faces[:,:], :]
    cm.save(stl_fname)

def rho2mesh(rho, mesh_fname, spacing=(2.,2.,2.)):
    # This method works, but it's boxy
    # Use gen_mesh_from_voxels
    vprnt("Generate boxy mesh from voxels")
    vmesh = femg.gen_mesh_from_voxels(rho, spacing)
    vmesh.write(mesh_fname)
    vprnt(vmesh)
    vprnt(f"Saved to {mesh_fname}")

def stl2msh(fname_stl, fname_msh):
    """ Convert an STL to a GMSH msh file.
    This function calls GMSH via subprocess because the python API wasn't
        working for me.
    """
    geo = f"Merge \"{fname_stl}\";\n"
    # TODO: This currently assumes one volume. It could probably be rewritten
    #   to handle an arbitrary number. See stl2msh_py.
    geo += "Surface Loop(1) = {1};\n"
    geo += "Volume(1) = {1};\n"
    geo += "Mesh 3;\n"
    geo += f"Save \"{fname_msh}\";"
    # Save this string to a geo file
    fname_geo = "tmp_stl2msh.geo"
    with open(fname_geo, "w") as fg:
        fg.write(geo)
    # Run that geo file
    #subprocess.run(["gmsh", fname_geo, "-3"])
    subprocess.run(["gmsh", fname_geo, "-"])

def stl2msh_py(fname_stl, fname_msh):
    # Using python GMSH interface.
    # https://gitlab.onelab.info/gmsh/gmsh/blob/gmsh_4_11_1/api/gmsh.py
    # For some reason this way wasn't working.

    assert False, "NOT IMPLEMENTED"

    gmsh.initialize()
    # Import STL
    gmsh.merge(fname_stl)
    # Define options
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(1, "F", "1000")
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    # Create physical surfaces
    gmsh.model.mesh.generate(3)
    #surf_tags, _ = gmsh.model.getEntities(2)
    # TODO: this part isn't working.
    surf_tags = [entity[1] for entity in gmsh.model.getEntities(2)]
    loop_tag = gmsh.model.geo.addSurfaceLoop(surf_tags)
    vol_tag = gmsh.model.geo.addVolume([loop_tag])
    gmsh.model.addPhysicalGroup(3, [vol_tag], 1)
    #for tag in surf_tags:
    #    gmsh.model.addPhysicalGroup(2, [tag], tag)
    # create the volume mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(fname_msh)
    gmsh.finalize()

    """ OLD
    # https://gmsh.info/doc/texinfo/gmsh.html#t13
    gmsh.initialize()
    gmsh.merge("tmp_sph.stl")
    #gmsh.model.mesh.createGeometry()
    gmsh.model.reparametrizeOnSurface(2, 1, [], 1)
    # I'm not sure if this next part will work
    gmsh.model.geo.addSurfaceLoop([1])
    gmsh.model.geo.addVolume([1])
    gmsh.model.mesh.optimize() # Optional
    gmsh.write("tmp_sph.msh")
    """

def msh2sfepy(fname_msh, fname_sfepy):
    """ Convert gmsh .msh to SfePy .mesh or .vtk
    This function calls sfepy-convert via subprocess

    TODO: Is there an option to suppress output? Same for gmsh

    E.g. $ sfepy-convert -m -d 3 hyprb.msh hyprb.vtk
    """
    subprocess.run(["sfepy-convert", "-d 3", fname_msh, fname_sfepy])

def cube_isosurf():
    # Sinusoidal --> bubbles
    h = 1
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-10, 10)
    x = np.arange(x_lims[0], x_lims[1], h)
    y = np.arange(y_lims[0], y_lims[1], h)
    z = np.arange(z_lims[0], z_lims[1], h)
    X, Y, Z = np.meshgrid(x, y, z)
    rho = np.maximum(np.abs(X), np.abs(Y), np.abs(Z))
    # # Modify density function to deal with faces
    # rho[0,:,:] = 0
    # rho[-1,:,:] = 0
    # rho[:,0,:] = 0
    # rho[:,-1,:] = 0
    # rho[:,:,0] = 1 # Assuming Z is the third index... CHECK TODO
    # rho[:,:,-1] = 1
    # Modify density function to deal with faces
    xmin, xmax = (-5,5)
    ymin, ymax = (-5,5)
    zmin, zmax = (-5,5)
    rho[(X <= xmin) or (X >= xmax)] = 0
    rho[(Y <= ymin) or (Y >= ymax)] = 0
    rho[(Z <= zmin) or (Z >= zmax)] = 1
    # Threshold the density function with an isosurface
    rho_cutoff = 1
    verts, faces, normals, values = skim.marching_cubes(rho, rho_cutoff)
    # NOTE: The units for position here are all integer indexes... I think.
    #   May need to convert back -- TODO
    # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes
    # There's an option for that (spacing). Also play with the other options.
    isosurf = [verts, faces, normals, values]
    return isosurf

def x0_bubbles():
    # Sinusoidal --> bubbles
    h = 1
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-10, 10)
    x = np.arange(x_lims[0], x_lims[1], h)
    y = np.arange(y_lims[0], y_lims[1], h)
    z = np.arange(z_lims[0], z_lims[1], h)
    X, Y, Z = np.meshgrid(x, y, z)
    k = .25*np.pi # angular wavenumber
    rho = (np.cos(X*k) + np.cos(Y*k) + np.cos(Z*k))/3

    return rho.flatten()

def bubbles_isosurf():
    # Sinusoidal --> bubbles
    h = 1
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-9, 9)
    x = np.arange(x_lims[0], x_lims[1], h)
    y = np.arange(y_lims[0], y_lims[1], h)
    z = np.arange(z_lims[0], z_lims[1], h)
    X, Y, Z = np.meshgrid(x, y, z)
    k = .25*np.pi # angular wavenumber
    rho = (np.cos(X*k) + np.cos(Y*k) + np.cos(Z*k))/3

    return rho2isosurf(rho)

def sphere_isosurf():
    # Try to make a mesh from voxel data
    h = 1
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-10, 10)
    x = np.arange(x_lims[0], x_lims[1], h)
    y = np.arange(y_lims[0], y_lims[1], h)
    z = np.arange(z_lims[0], z_lims[1], h)
    X, Y, Z = np.meshgrid(x, y, z)
    R2 = X**2 + Y**2 + Z**2
    r0 = 8
    # Voxel boolean: filled or not
    data = R2 < r0**2 # NOT USED ANYMORE
    verts, faces, normals, values = skim.marching_cubes(R2, r0**2)
    #return skim.marching_cubes(R2, r0**2)
    isosurf = [verts, faces, normals, values]
    return isosurf

def cylinder_isosurf():
    # Try to make a mesh from voxel data
    h = 1
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-40, 40)
    x = np.arange(x_lims[0], x_lims[1], h)
    y = np.arange(y_lims[0], y_lims[1], h)
    z = np.arange(z_lims[0], z_lims[1], h)
    X, Y, Z = np.meshgrid(x, y, z)
    R2 = X**2 + Y**2
    r0 = 8
    # data = R2 < r0**2 # boolean filled or not
    #verts, faces, normals, values = skim.marching_cubes(R2, r0**2)
    verts, faces, normals, values = skim.marching_cubes(R2, r0**2)
    #return skim.marching_cubes(R2, r0**2)
    isosurf = [verts, faces, normals, values]
    return isosurf

def x0_hyperboloid(dim=(10,10,10)):
    """ Generate an x-vector for a hyperboloid as an initial guess
    """
    h = 1
    x_lims = (-10, 10)
    y_lims = (-10, 10)
    z_lims = (-10, 10)
    # This method places a point at either limit. Alternatively, I could place
    #   each point at the center of a box -- TODO
    Lx = max(x_lims) - min(x_lims)
    Ly = max(x_lims) - min(x_lims)
    Lz = max(x_lims) - min(x_lims)
    hx = Lx / (dim[0] - 1)
    hy = Ly / (dim[1] - 1)
    hz = Lz / (dim[2] - 1)
    x = np.arange(x_lims[0], x_lims[1]+hx/2, hx)
    y = np.arange(y_lims[0], y_lims[1]+hy/2, hy)
    z = np.arange(z_lims[0], z_lims[1]+hz/2, hz)
    X, Y, Z = np.meshgrid(x, y, z)
    a = Lx / 4 # a = skirt radius
    c = Lz / 4 * 1.25
    # https://mathworld.wolfram.com/One-SheetedHyperboloid.html
    hyperboloid = X**2/a**2 + Y**2/a**2 - Z**2/c**2
    # Negative because we want material inside, not outside
    rho = 1.5 - hyperboloid # assuming rho threshold = 0.5
    # print(rho)
    return rho.flatten()

if __name__ == "__main__":

    print("TESTING MESHING")

    # Test meshing a hyperboloid
    x0 = x0_hyperboloid()
    #x0 = x0_bubbles()
    x2mesh(x0, "hyprbA", out_format="mesh")

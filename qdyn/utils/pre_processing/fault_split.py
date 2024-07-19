import numpy as np
import noise
from scipy.ndimage import uniform_filter

def create_double_fault(input_mesh, displacement_x = 0, displacement_y = 0, displacement_z = 0, len_vs = 0.5e3, noise_level = 0, smoothing_window = 500, b_fault_input = 0.0042):

    mesh = input_mesh

    if displacement_x < 0 or displacement_y < 0 or displacement_z < 0:
        raise ValueError("Error: All input displacements must be positive values")

    # Divide the input fault into two discrete equal faults

    Nw = mesh.set_dict["NW"]
    Nx = mesh.set_dict["NX"]

    split = Nw // 2

    X = mesh.mesh_dict["X"].reshape((Nw, Nx))
    Y = mesh.mesh_dict["Y"].reshape((Nw, Nx))
    Z = mesh.mesh_dict["Z"].reshape((Nw, Nx))

    dX = displacement_x
    dY = displacement_y
    dW = mesh.set_dict["W"] / 2
    dZ_ = dW * np.sin(np.radians(mesh.set_dict["DIP_W"]))
    dZ = displacement_z + dZ_
    d_Z = displacement_z

    z_cut_point = input_mesh.set_dict["Z_CORNER"] + dZ_

    mesh.mesh_dict["N_FAULTS"] = 2
    mesh.mesh_dict["FAULT_LABEL"] = np.where(input_mesh.mesh_dict["Z"] > z_cut_point, 2, 1)   # ensures labels for the two new faults are correct

    # Displace the mesh elements according to input displacements, so two faults are created at the specified offsets

    mesh.mesh_dict["X"] = np.where(mesh.mesh_dict["Z"]>z_cut_point, mesh.mesh_dict["X"]+dX, mesh.mesh_dict["X"])
    mesh.mesh_dict["Y"] = np.where(mesh.mesh_dict["Z"]>z_cut_point, mesh.mesh_dict["Y"]+dY-(dZ/np.tan(np.radians(mesh.set_dict["DIP_W"]))), mesh.mesh_dict["Y"])
    mesh.mesh_dict["Z"] = np.where(mesh.mesh_dict["Z"]>z_cut_point, mesh.mesh_dict["Z"]-dZ, mesh.mesh_dict["Z"])

    xmin = mesh.mesh_dict["X"].min()
    xmax = mesh.mesh_dict["X"].min() + mesh.set_dict["L"]
    zmin = mesh.mesh_dict["Z"].min()
    zmax = mesh.mesh_dict["Z"].min() + dZ_

    x_vw_min = xmin + len_vs
    x_vw_max = xmax - len_vs
    z_vw_min = zmin + len_vs * np.sin(np.radians(mesh.mesh_dict["DIP_W"]))   # accounts for the dip, and fault aspect ratio
    z_vw_max = zmax - len_vs * np.sin(np.radians(mesh.mesh_dict["DIP_W"]))

    ivw = np.where((mesh.mesh_dict["X"]>=x_vw_min) & (mesh.mesh_dict["X"]<=x_vw_max) & (mesh.mesh_dict["Z"]>=z_vw_min) & (mesh.mesh_dict["Z"]<=z_vw_max))   # ivw = in velocity weakening
    ivs = np.where(((mesh.mesh_dict["X"]<x_vw_min) | (mesh.mesh_dict["X"]>x_vw_max)) | ((mesh.mesh_dict["Z"]<z_vw_min) | (mesh.mesh_dict["Z"]>z_vw_max)))   # ivs = in velocity strengthening
    ivs_x1 = np.where(((mesh.mesh_dict["X"]<x_vw_min) | (mesh.mesh_dict["X"]>x_vw_max)))   # ivs_x = in velocity strengthening, extended along z direction
    ivs_x2 = np.where(((mesh.mesh_dict["X"]<(x_vw_min + dX)) | (mesh.mesh_dict["X"]>(x_vw_max + dX))))   # ivs_x = in velocity strengthening, extended along z direction
    ivs_z1 = np.where(((mesh.mesh_dict["Z"]<z_vw_min) | (mesh.mesh_dict["Z"]>z_vw_max)))   # ivs_z = in velocity strengthening, extended along x direction
    ivs_z2 = np.where(((mesh.mesh_dict["Z"]<(z_vw_min + d_Z)) | (mesh.mesh_dict["Z"]>(z_vw_max + d_Z))))   # ivs_z = in velocity strengthening, extended along x direction

    # assign new b value outside the asperity on all three permutations
    b_asperity = mesh.set_dict["SET_DICT_RSF"]["B"]
    b_fault = b_fault_input

    b_orig = mesh.mesh_dict["B"].copy()
    b_smx1 = mesh.mesh_dict["B"].copy()
    b_smx2 = mesh.mesh_dict["B"].copy()
    b_smz1 = mesh.mesh_dict["B"].copy()
    b_smz2 = mesh.mesh_dict["B"].copy()

    b_orig[ivs]  = b_fault
    b_smx1[ivs_x1] = b_fault
    b_smx2[ivs_x2] = b_fault
    b_smz1[ivs_z1] = b_fault
    b_smz2[ivs_z2] = b_fault

    window_size = smoothing_window  # Window size for smoothing in meters

    # Calculate the number of grid points corresponding to the window size
    x_unique = np.unique(mesh.mesh_dict["X"])
    z_unique = np.unique(mesh.mesh_dict["Z"])
    dx = x_unique[1] - x_unique[0]  # Grid spacing in x-direction
    dz = z_unique[0] - z_unique[1]  # Grid spacing in z-direction
    window_size_x = int(window_size / dx)
    window_size_z = int(window_size / dz)

    # reshape B array for Z direction
    b_0 = (b_orig).reshape((Nw, Nx))
    b_2 = (b_smz2).reshape((Nw, Nx))
    b_1 = (b_smz1).reshape((Nw, Nx))
    b1_zd = b_2[:split].T # fault 1
    b2_zd = b_1[split:].T # fault 2

    # Apply smoothing filter for Z direction
    b1_smoothed = uniform_filter(b1_zd, size=(window_size_z, window_size_x), mode='constant')
    b2_smoothed = uniform_filter(b2_zd, size=(window_size_z, window_size_x), mode='constant')

    b_smoothed = b_0.copy()
    b_smoothed[:split] = b1_smoothed.T
    b_smoothed[split:] = b2_smoothed.T

    b_smoothed_T = b_smoothed.ravel()

    # reshape B array for X direction
    b_xd1 = (b_smx1).reshape((Nw, Nx))
    b_xd2 = (b_smx2).reshape((Nw, Nx))
    b1_xd = b_xd1[:split]
    b2_xd = b_xd2[split:]

    # Apply smoothing filter for X direction
    b1_smoothed_xd = uniform_filter(b1_xd, size=(window_size_z, window_size_x), mode='constant')
    b2_smoothed_xd = uniform_filter(b2_xd, size=(window_size_z, window_size_x), mode='constant')

    b_smoothed_xd = b_xd1.copy()
    b_smoothed_xd[:split] = b1_smoothed_xd
    b_smoothed_xd[split:] = b2_smoothed_xd
    b_smoothed_xd_flat = b_smoothed_xd.ravel()

    # to get a smooth transition on all sides, merge the two smoothing directions and take the minimum from each
    mesh.mesh_dict["B"] = np.minimum(b_smoothed_xd_flat, b_smoothed_T)
    mesh.mesh_dict["B"][mesh.mesh_dict["B"] < b_fault] = b_fault  # removes patches of anomalously low b, caused by smoothing over a finite mesh

    # apply noise

    # function to create perlin noise
    def apply_perlin_noise(array, amplitude):
        noisy_array = np.empty_like(array)
        for i in range(len(array)):
            perlin_noise = noise.pnoise1(i * 0.1)  # Scale the index by a constant factor
            noisy_array[i] = array[i] + (perlin_noise * amplitude)
        return noisy_array

    # apply noise
    b = mesh.mesh_dict["B"].copy()
    amplitude = noise_level

    b_noisy1 = apply_perlin_noise(b[split:], amplitude)
    b_noisy2 = apply_perlin_noise(b[:split], amplitude)

    b[split:] = b_noisy1
    b[:split] = b_noisy2

    # Override the default mesh values
    mesh.mesh_dict["B"] = b

    return(mesh)

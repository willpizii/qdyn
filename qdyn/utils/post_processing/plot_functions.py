"""
Useful functions for post-processing (plotting) 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import HTML
import matplotlib.cm as cm
import pandas as pd


t_day = 3600 * 24.0
t_year = 365 * t_day
figsize = (9, 8)


def timeseries(ot, ot_vmax, tmin=None, tmax=None): # timeseries plots supporting datasets with multiple outputs

    if isinstance(ot, pd.DataFrame):  # fallback case if a passed input is only one series
        ot = [ot]
    # assign tmin and tmax of the simulation if None
    if tmin == None:
        tmin = ot[0]['t'].min()
    if tmax == None:
        tmax = ot[0]['t'].max()

    # check if tmin and tmax are bounded correctly
    if tmin < ot[0]['t'].min():
        print('tmin is smaller than min t of simulation')
        return
    if tmax > ot[0]['t'].max():
        print('tmax is larger than max t of simulation')
        return

    # filter the data based on tmin and tmax
    ot_sample = [df[(df['t'] / t_year >= tmin) & (df['t'] / t_year <= tmax)] for df in ot]
    ot_vmax_sample = ot_vmax[(ot_vmax['t'] / t_year >= tmin) & (ot_vmax['t'] / t_year <= tmax)]

    plt.rcParams.update(plt.rcParamsDefault)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=figsize)

    for i in range(len(ot_sample)):

        axes[0].plot(ot_sample[i]["t"] / t_year, ot_sample[i]["tau"])
        axes[0].set_ylabel("tau [Pa]")

        axes[1].plot(ot_sample[i]["t"] / t_year, ot_sample[i]["theta"])
        axes[1].set_ylabel("state [s]")

        axes[2].plot(ot_sample[i]["t"] / t_year, ot_sample[i]["sigma"], label=f"Fault {i+1}")
        axes[2].set_ylabel("sigma [Pa]")
        axes[2].legend()

    axes[3].plot(ot_vmax_sample["t"] / t_year, ot_vmax_sample["v"])
    axes[3].set_ylabel("max v [m/s]")
    axes[3].set_xlabel("time [yr]")
    axes[3].set_yscale("log")

    plt.tight_layout()
    plt.show()

def slip_profile(ox, warm_up=0, orientation="horizontal"):
    x_unique = ox["x"].unique()
    sort_inds = np.argsort(x_unique)
    x_unique = x_unique[sort_inds]
    t_vals = np.sort(ox["t"].unique())

    if warm_up > t_vals.max():
        print("Warm-up time > simulation time!")
        return

    ind_warmup = np.where(t_vals >= warm_up)[0][0]

    Nx = len(x_unique)
    Nt = len(t_vals) - 1

    slice = np.s_[Nx * ind_warmup:Nx * Nt]
    data_shape = (Nt - ind_warmup, Nx)
    x = ox["x"][slice].values.reshape(data_shape)[:, sort_inds]
    slip = ox["slip"][slice].values.reshape(data_shape)[:, sort_inds]
    v = ox["v"][slice].values.reshape(data_shape)[:, sort_inds]

    slip -= slip[0]
    t = t_vals[ind_warmup:-1]
    t -= t[0]

    fig = plt.figure(figsize=figsize)

    if orientation == "horizontal":
        CS = plt.contourf(x, slip, np.log10(v), levels=200, cmap="magma")
        plt.xlabel("position [m]")
        plt.ylabel("slip [m]")
        CB = plt.colorbar(CS, orientation="horizontal")
        CB.ax.set_title("slip rate [m/s]")
    elif orientation == "vertical":
        x -= x.min()
        CS = plt.contourf(slip, x * 1e-3, np.log10(v), levels=200, cmap="magma")
        plt.ylabel("depth [km]")
        plt.xlabel("slip [m]")
        plt.gca().invert_yaxis()
        CB = plt.colorbar(CS, orientation="horizontal")
        CB.ax.set_title("slip rate [m/s]")
    else:
        print("Keyword 'orientation=%s' not recognised" % orientation)
        plt.close()
        return
    plt.tight_layout()
    plt.show()

def slip_profile_new(ox, dip, warm_up=0, orientation="horizontal", figsize=figsize):
    """
    Plot slip profile along the fault trace.

    Parameters
    ----------
    ox : pandas DataFrame
        DataFrame containing ox output. 
        If fault is 0D or 1D --> ox=p.ox
        If fault is 2D, the x or z coordinate has to be provided
        --> for vertical cross-section: ox = p.ox[(p.ox["x"] == x_pos)] where x_pos is the x coordinate of the cross-section
        --> for horizontal cross-section: ox = p.ox[(p.ox["z"] == z_pos)] where z_pos is the z coordinate of the cross-section
        If there are multiple faults, the fault label has to be provided
        --> ox[(ox["fault_label"] == 1)]
        Example for slip profile of a vertical cross-section of Fault 1:
        p.ox[(p.ox["fault_label"] == 1) & (p.ox["x"] == x_pos)]
    dip : float
        Dip angle in degrees
    warm_up : float, optional
        Time in seconds to disregard in the plot (default is 0)
    orientation : str, optional
        Plot/cross section orientation. Can be 'horizontal' (default) or 'vertical'
    figsize : tuple, optional
        Figure size. Default (9,8)

    """
    
    # Check orientation parameter
    if orientation not in ["horizontal", "vertical"]:
        print("Error: orientation parameter must be either 'horizontal' or 'vertical'")
        return
    
    # Determine coordinate to plot along
    if orientation=="horizontal":
        coord="x"
    else:
        coord="z"
    
    # Get unique coordinates and sort them
    x_unique = ox[coord].unique()
    sort_inds = np.argsort(x_unique)
    x_unique = x_unique[sort_inds]
    
    # Get unique times and sort them
    t_vals = np.sort(ox["t"].unique())
    
    # Check that warm-up time is not greater than simulation time
    if warm_up > t_vals.max():
        print("Warm-up time > simulation time!")
        return
    
    # Determine index for warm-up time
    ind_warmup = np.where(t_vals >= warm_up)[0][0]
    
    # Get the number of unique coordinates and number of time steps
    Nx = len(x_unique)
    Nt = len(t_vals) - 1
    
    # Define the slice and shape for the data
    slice = np.s_[Nx * ind_warmup:Nx * Nt]
    data_shape = (Nt - ind_warmup, Nx)
    
    # Get the data values and reshape them
    x = ox[coord][slice].values.reshape(data_shape)[:, sort_inds]
    slip = ox["slip"][slice].values.reshape(data_shape)[:, sort_inds]
    v = ox["v"][slice].values.reshape(data_shape)[:, sort_inds]
    
    # Adjust slip values to start at zero
    slip -= slip[0]
    
    # Adjust time values to start at zero
    t = t_vals[ind_warmup:-1]
    t -= t[0]

    # Calculate downdip distance for vertical profile
    if orientation=="vertical":
        ddd = x / np.sin(np.deg2rad(dip))
    
    # Create figure and plot the slip profile
    fig = plt.figure(figsize=figsize)

    if orientation == "horizontal":
        CS = plt.contourf(x, slip, np.log10(v), levels=200, cmap="magma")
        plt.xlabel("position [m]")
        plt.ylabel("slip [m]")
        CB = plt.colorbar(CS, orientation="horizontal")
        CB.ax.set_title("slip rate [m/s]")
    elif orientation == "vertical":
        ddd -= ddd.min()
        CS = plt.contourf(slip, ddd * 1e-3, np.log10(v), levels=200, cmap="magma")
        plt.ylabel("downdip distance [km]")
        plt.xlabel("slip [m]")
        plt.gca().invert_yaxis()
        CB = plt.colorbar(CS, orientation="horizontal")
        CB.ax.set_title("slip rate [m/s]")
    else:
        print("Keyword 'orientation=%s' not recognised" % orientation)
        plt.close()
        return

    plt.tight_layout()
    plt.show()



def timestep_profile(ox, dip, warm_up=0, orientation="horizontal", figsize=figsize):
    """
    Plot timestep profile along the fault trace with slip rate cotouring.

    Parameters
    ----------
    ox : pandas DataFrame
        DataFrame containing ox output. 
        If fault is 0D or 1D --> ox=p.ox
        If fault is 2D, the x or z coordinate has to be provided
        --> for vertical cross-section: ox = p.ox[(p.ox["x"] == x_pos)] where x_pos is the x coordinate of the cross-section
        --> for horizontal cross-section: ox = p.ox[(p.ox["z"] == z_pos)] where z_pos is the z coordinate of the cross-section
        If there are multiple faults, the fault label has to be provided
        --> ox[(ox["fault_label"] == 1)]
        Example for slip profile of a vertical cross-section of Fault 1:
        p.ox[(p.ox["fault_label"] == 1) & (p.ox["x"] == x_pos)]
    dip : float
        Dip angle in degrees
    warm_up : float, optional
        Time in seconds to disregard in the plot (default is 0)
    orientation : str, optional
        Plot/cross section orientation. Can be 'horizontal' (default) or 'vertical'
    figsize : tuple, optional
        Figure size. Default (9,8)
    
    
    Example
    -------
    To plot an horizontal profile in the middle of a 2D dipping fault with label 1 from a model with two faults:
    >>>> timestep_profile(p.ox[(p.ox["z"] == z_pos) &  (p.ox["fault_label"] == 1)], dip=60, warm_up=50*t_yr, orientation="horizontal")

    """
    
    # Check orientation parameter
    if orientation not in ["horizontal", "vertical"]:
        print("Error: orientation parameter must be either 'horizontal' or 'vertical'")
        return
    
    # Determine coordinate to plot along
    if orientation=="horizontal":
        coord="x"
    else:
        coord="z"
    
    # Get unique coordinates and sort them
    x_unique = ox[coord].unique()
    sort_inds = np.argsort(x_unique)
    x_unique = x_unique[sort_inds]
    
    # Get unique times and sort them
    t_vals = np.sort(ox["t"].unique())
    
    # Check that warm-up time is not greater than simulation time
    if warm_up > t_vals.max():
        print("Warm-up time > simulation time!")
        return
    
    # Determine index for warm-up time
    ind_warmup = np.where(t_vals >= warm_up)[0][0]
    
    # Get the number of unique coordinates and number of time steps
    Nx = len(x_unique)
    Nt = len(t_vals) - 1
    
    # Define the slice and shape for the data
    slice = np.s_[Nx * ind_warmup:Nx * Nt]
    data_shape = (Nt - ind_warmup, Nx)
    
    # Get the data values and reshape them
    x = ox[coord][slice].values.reshape(data_shape)[:, sort_inds]
    timestep = ox["step"][slice].values.reshape(data_shape)[:, sort_inds]
    v = ox["v"][slice].values.reshape(data_shape)[:, sort_inds]
    
    # max and min values
    
    
    # Adjust timestep values to start at zero
    # slip -= slip[0]
    timestep -= timestep[0]
    
    # Adjust time values to start at zero
    t = t_vals[ind_warmup:-1]
    t -= t[0]

    # Calculate downdip distance for vertical profile
    if orientation=="vertical":
        ddd = x / np.sin(np.deg2rad(dip))
    
    # Create figure and plot the slip profile
    fig = plt.figure(figsize=figsize)

    if orientation == "horizontal":
        CS = plt.contourf(timestep, x, np.log10(v), levels=100, cmap="magma")
        plt.ylabel("position [m]")
        # plt.ylabel("slip [m]")
        plt.xlabel("Timestep")
        CB = plt.colorbar(CS, orientation="horizontal")
        CB.ax.set_title("slip rate [m/s]")
    elif orientation == "vertical":
        ddd -= ddd.min()
        CS = plt.contourf(ddd * 1e-3, timestep, np.log10(v), levels=100, cmap="magma")
        plt.xlabel("downdip distance [km]")
        # plt.xlabel("slip [m]")
        plt.ylabel("Timestep")
        plt.gca().invert_yaxis()
        CB = plt.colorbar(CS, orientation="horizontal")
        CB.ax.set_title("slip rate [m/s]")
    else:
        print("Keyword 'orientation=%s' not recognised" % orientation)
        plt.close()

    plt.tight_layout()
    plt.show()
        
    return fig

# CRP: Experimental!
def timestep_profile_all(ox, dip, val, sigma_0=None, warm_up=0, orientation="horizontal", figsize=figsize):

    """
    Plot timestep profile along the fault trace with a value from a col of ox as contouring.

    Additional libraries
    --------
    cmcrameri

    Parameters
    ----------
    ox : pandas DataFrame
        DataFrame containing ox output.
        If the fault is 0D or 1D --> ox = p.ox
        If the fault is 2D, the x or z coordinate has to be provided:
        - For a vertical cross-section: ox = p.ox[(p.ox["x"] == x_pos)], where x_pos is the x coordinate of the cross-section.
        - For a horizontal cross-section: ox = p.ox[(p.ox["z"] == z_pos)], where z_pos is the z coordinate of the cross-section.
        If there are multiple faults, the fault label has to be provided:
        - ox[(ox["fault_label"] == 1)]
        Example for plotting a slip profile of a vertical cross-section of Fault 1:
        p.ox[(p.ox["fault_label"] == 1) & (p.ox["x"] == x_pos)]
    dip : float
        Dip angle in degrees.
    val : str
        The value to plot. It can be one of the following:
        - "v" for slip rate.
        - "dsigma" for change in stress (requires sigma_0 parameter).
        - "theta" for the angle parameter.
        - "tau" for shear stress.
        - "tau_dot" for shear stress rate.
        - "sigma" for normal stress.
        - "tau_sigma" for the ratio of shear stress to normal stress.
        - "slip" for cumulative slip.
    sigma_0 : float, optional
        Initial stress value. Required if val is "dsigma". Default is None.
    warm_up : float, optional
        Time in seconds to disregard in the plot. Default is 0.
    orientation : str, optional
        Plot/cross-section orientation. Can be "horizontal" (default) or "vertical" (for 2D fault).
        Note that it will raise an error if trying to plot a "vertical" cross section in a fault other than in 2D.
    figsize : tuple, optional
        Figure size. Default is (9, 8).

    Examples
    --------
    To plot a horizontal profile in the middle of a 2D dipping fault with label 1 from a model with two faults:
    >>>> timestep_profile_all(p.ox[(p.ox["z"] == z_pos) & (p.ox["fault_label"] == 1)], dip=60, val="v", warm_up=50*t_yr, orientation="horizontal")

    """
    import matplotlib as mpl
    import cmcrameri.cm as cmc

    # Check orientation parameter
    if orientation not in ["horizontal", "vertical"]:
        print("Error: orientation parameter must be either 'horizontal' or 'vertical'")
        return

    # Determine coordinate to plot along
    if orientation=="horizontal":
        coord="x"
    else:
        coord="z"

    # Get unique coordinates and sort them
    x_unique = ox[coord].unique()
    sort_inds = np.argsort(x_unique)
    x_unique = x_unique[sort_inds]

    # Get unique times and sort them
    t_vals = np.sort(ox["t"].unique())

    # Check that warm-up time is not greater than simulation time
    if warm_up > t_vals.max():
        raise ValueError("Warm-up time > simulation time!")

    # Determine index for warm-up time
    ind_warmup = np.where(t_vals >= warm_up)[0][0]

    # Get the number of unique coordinates and number of time steps
    Nx = len(x_unique)
    Nt = len(t_vals) - 1

    # Define the slice and shape for the data
    slice = np.s_[Nx * ind_warmup:Nx * Nt]
    data_shape = (Nt - ind_warmup, Nx)

    # Get the data values and reshape them
    x = ox[coord][slice].values.reshape(data_shape)[:, sort_inds]
    timestep = ox["step"][slice].values.reshape(data_shape)[:, sort_inds]

    if val=="v":
        v = ox["v"][slice].values.reshape(data_shape)[:, sort_inds]
    elif val=="dsigma":
        # check sigma
        if sigma_0 ==None:
            raise ValueError("You have to set the value for sigma_0")
        dsigma = sigma_0 - ox["sigma"][slice].values.reshape(data_shape)[:, sort_inds]
    elif val=="theta":
        theta = ox["theta"][slice].values.reshape(data_shape)[:, sort_inds]
    elif val=="tau":
        tau = ox["tau"][slice].values.reshape(data_shape)[:, sort_inds]
    elif val=="tau_dot":
        tau_dot = ox["tau_dot"][slice].values.reshape(data_shape)[:, sort_inds]
    elif val=="sigma":
        sigma = ox["sigma"][slice].values.reshape(data_shape)[:, sort_inds]
    elif val=="tau_sigma":
        # check sigma
        if sigma_0 ==None:
            raise ValueError("You have to set the value for sigma_0")
        tau_sigma = ox["tau"][slice].values.reshape(data_shape)[:, sort_inds] / sigma_0
    elif val=="slip":
        slip = ox["slip"][slice].values.reshape(data_shape)[:, sort_inds]
    else:
        raise ValueError("val not included in ox")

    # Adjust timestep values to start at zero
    # slip -= slip[0]
    timestep -= timestep[0]

    # Adjust time values to start at zero
    t = t_vals[ind_warmup:-1]
    t -= t[0]

    # Calculate downdip distance for vertical profile
    if orientation=="vertical":
        ddd = x / np.sin(np.deg2rad(dip))

    # Create figure and plot the slip profile
    fig, ax = plt.subplots(figsize=figsize)

    # Special colorscales
    if val == "v":
        # make a colormap that has creep and dyn clearly delineated
        # make a colormap that has land and ocean clearly delineated and of the
        # same length (256 + 256)
        colors_creep = cmc.batlowW(np.linspace(0, 0.4, 128))
        colors_dyn = cmc.batlowW(np.linspace(0.6, 1, 128))
        all_colors = np.vstack((colors_creep, colors_dyn))
        vcmap = mpl.colors.LinearSegmentedColormap.from_list(
            'vc_map', all_colors)

        # norm
        vmin = np.log10(1e-14)
        vmax = np.log10(1e1)
        vnorm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=np.log10(1e-2), vmax=vmax)

        #ticks (adjust number of elements according to vmin and vmax)
        ticks = np.linspace(vmin, vmax, 16, endpoint=True)

        # Define the contour levels with a break at v=1e-2
        levels = [-np.inf, np.log10(1e-2), np.inf]


    elif val == "dsigma":
        vcmap = cmc.vik
        vnorm = mpl.colors.CenteredNorm()

        #ticks (adjust number of elements according to vmin and vmax) --> not used for now
        dsigma_ox = sigma_0 - ox["sigma"]

        vmax = np.max([abs(max(dsigma_ox)), abs(max(dsigma_ox))])
        vmin = -vmax

        ticks = np.linspace(vmin, vmax, 10, endpoint=True)


    if orientation == "horizontal":
        if val == "v":
            CS = plt.contourf(timestep, x, np.log10(v), levels=200, cmap=vcmap, norm=vnorm)
            CB = plt.colorbar(plt.cm.ScalarMappable(cmap=vcmap, norm=vnorm), ax=ax, orientation="horizontal", extend='both', ticks=ticks, pad=0.2)
            CB.ax.set_title("log v [m/s]")
        elif val=="dsigma":
            CS = plt.contourf(timestep, x, dsigma/1e6, levels=200, cmap=vcmap, norm=vnorm)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("dsigma [MPa]")
        elif val=="sigma":
            CS = plt.contourf(timestep, x, sigma/1e6, levels=200, cmap=cmc.batlow)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("sigma [MPa]")
        elif val=="tau":
            CS = plt.contourf(timestep, x, tau/1e6, levels=200, cmap=cmc.batlow)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("tau [MPa]")
        elif val == "tau_sigma":
            CS = plt.contourf(timestep, x, tau_sigma, levels=200, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("tau/sigma")
        elif val=="theta":
            CS = plt.contourf(timestep, x, theta, levels=200, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("theta")
        elif val == "tau_dot":
            CS = plt.contourf(timestep, x, tau_dot, levels=200, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("tau_dot")
        elif val == "slip":
            CS = plt.contourf(timestep, x, slip, levels=20, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("cumulative slip (m)")

        plt.ylabel("position (m)")
        plt.xlabel("Timestep")

    elif orientation == "vertical":
        ddd -= ddd.min()

        if val == "v":
            CS = plt.contourf(ddd, timestep, levels=200, cmap=vcmap, norm=vnorm)
            CB = plt.colorbar(plt.cm.ScalarMappable(cmap=vcmap, norm=vnorm), ax=ax, orientation="horizontal", extend='both', ticks=ticks, pad=0.2)
            CB.ax.set_title("log v [m/s]")
        elif val=="dsigma":
            CS = plt.contourf(ddd, timestep, dsigma/1e6, levels=200, cmap=vcmap, norm=vnorm)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("dsigma [MPa]")
        elif val=="sigma":
            CS = plt.contourf(ddd, timestep, sigma/1e6, levels=200, cmap=cmc.batlow)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("sigma [MPa]")
        elif val=="tau":
            CS = plt.contourf(ddd, timestep, tau/1e6, levels=200, cmap=cmc.batlow)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("tau [MPa]")
        elif val == "tau_sigma":
            CS = plt.contourf(ddd, timestep, tau_sigma, levels=200, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("tau/sigma")
        elif val=="theta":
            CS = plt.contourf(ddd, timestep, theta, levels=200, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("theta")
        elif val == "tau_dot":
            CS = plt.contourf(ddd, timestep, tau_dot, levels=200, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("tau_dot")
        elif val == "slip":
            CS = plt.contourf(ddd, timestep, slip, levels=50, cmap=cmc.lipari)
            CB = plt.colorbar(CS, orientation="horizontal", pad=0.2)
            CB.ax.set_title("cumulative slip (m)")

        plt.xlabel("downdip distance (m)")
        plt.ylabel("Timestep")
        plt.gca().invert_yaxis()

    else:
        print("Keyword 'orientation=%s' not recognised" % orientation)
        plt.close()

    plt.tight_layout()
    plt.show()

    return fig





def animation_slip(ox, warm_up=0, orientation="horizontal"):

    x_unique = ox["x"].unique()
    sort_inds = np.argsort(x_unique)
    x_unique = x_unique[sort_inds]
    t_vals = np.sort(ox["t"].unique())

    if warm_up > t_vals.max():
        print("Warm-up time > simulation time!")
        return

    ind_warmup = np.where(t_vals >= warm_up)[0][0]

    Nx = len(x_unique)
    Nt = len(t_vals) - 1

    slice = np.s_[Nx * ind_warmup:Nx * Nt]
    data_shape = (Nt - ind_warmup, Nx)
    slip = ox["slip"][slice].values.reshape(data_shape)[:, sort_inds]
    v = ox["v"][slice].values.reshape(data_shape)[:, sort_inds]

    slip -= slip[0]
    t = t_vals[ind_warmup:-1]
    t -= t[0]

    plt.ioff()

    if orientation == "horizontal":
        fig, ax = plt.subplots(nrows=2, ncols=1, figsize=figsize)

        ax[0].set_xlim((x_unique.min(), x_unique.max()))
        ax[0].set_ylim((slip.min(), slip.max()))
        ax[0].set_ylabel("slip [m]")

        ax[1].set_xlim((x_unique.min(), x_unique.max()))
        ax[1].set_ylim((v.min(), v.max()))
        ax[1].set_yscale("log")
        ax[1].set_ylabel("slip rate [m/s]")
        ax[1].set_xlabel("position [m]")

        line1, = ax[0].plot([], [], lw=2)
        line2, = ax[1].plot([], [], lw=2)
        lines = (line1, line2)

        def init_plot():
            for line in lines:
                line.set_data([], [])
            return lines,

        def animate(i):
            line1.set_data(x_unique, slip[i])
            line2.set_data(x_unique, v[i])

    elif orientation == "vertical":
        x_unique -= x_unique.min()
        x_unique *= 1e-3
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        ax[0].set_ylim((x_unique.min(), x_unique.max()))
        ax[0].set_xlim((slip.min(), slip.max()))
        ax[0].set_ylabel("depth [km]")
        ax[0].set_xlabel("slip [m]")
        ax[0].invert_yaxis()
        ax[0].xaxis.tick_top()
        ax[0].xaxis.set_label_position("top")

        ax[1].set_ylim((x_unique.min(), x_unique.max()))
        ax[1].set_xlim((v.min(), v.max()))
        ax[1].set_xscale("log")
        ax[1].set_xlabel("slip rate [m/s]")
        ax[1].invert_yaxis()
        ax[1].xaxis.tick_top()
        ax[1].xaxis.set_label_position("top")

        line1, = ax[0].plot([], [], lw=2)
        line2, = ax[1].plot([], [], lw=2)
        lines = (line1, line2)

        def init_plot():
            for line in lines:
                line.set_data([], [])
            return lines,

        def animate(i):
            line1.set_data(slip[i], x_unique)
            line2.set_data(v[i], x_unique)

    anim = animation.FuncAnimation(fig, animate, init_func=init_plot,
                                   frames=len(t), interval=30, blit=True)
    plt.tight_layout()
    HTML(anim.to_html5_video())
    rc("animation", html="html5")
    plt.close()
    return anim

def plot_snapshot_3d(ox, set_dict, t_snapshot=0, prop="v", fault_labels = [1], scaling="Normalize"):
    
    """
    Plot a 3D snapshot.

    Parameters:
        ox (DataFrame): p.ox containing the snapshots.
        set_dict (dict): p.set_dict containing settings and properties.
        t_snapshot (float): The time value (s) for the desired snapshot (default is 0s).
        prop (str): The property to visualize (default is "v").
        fault_labels (list): A list of fault labels to filter the data (default is [1]).
        scaling (str): The scaling method for the colormap (default is "Normalize").

    Raises:
        ValueError: If an invalid list of fault labels or scaling option is provided.

    Returns:
        None: This function displays the 3D plot using Matplotlib.
        
    Note:
        It is possible to plot dsigma (sigma_0 - sigma) even if it's not originally included
        in the ox DataFrame, since the calculation is internally handled by this function.

    Example:
        To plot a 3D snapshot of fault label 2 with property "dsigma" using a normalized colormap:
        >>> plot_snapshot_3d(p.ox, p.set_dict, t_snapshot=1.0, fault_labels=[2], prop="dsigma", scaling="Normalize")
    """
    
    # Handle errors
    unique_labels = np.unique(ox["fault_label"])
    if not all(label in unique_labels for label in fault_labels):
        raise ValueError("Invalid list of fault labels")
                              
    # Index of elements of snapshot
    inds = (ox["t"] == t_snapshot) & (ox["fault_label"].isin(fault_labels))

    # filter snapshot
    mesh_snapshot = ox[inds]

    # Draw canvas
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True, subplot_kw={"projection": "3d"})

    cmap = cm.magma

    # Decorate axes
    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("depth [km]")
    ax.set_title("t =" + str(t_snapshot/t_year) + " yr\n" + "fault " + str(fault_labels), pad = 20)

    # Set initial viewing angle
    ax.set_aspect("equal")
    ax.view_init(elev=51, azim=-60)

    # Select quantities to plot
    x = mesh_snapshot["x"]
    y = mesh_snapshot["y"]
    z = mesh_snapshot["z"]
    if prop == "dsigma":
        sigma_0 = set_dict["SIGMA"]
        col = sigma_0 - mesh_snapshot["sigma"]
    elif prop == "v":
        col = np.log10(mesh_snapshot[prop])
    else:
        col = mesh_snapshot[prop]

        
    # Colour scale normalisation
    if prop == "dsigma":
        cmap = cm.bwr
    else:
        cmap = cm.magma
    
    if scaling == "Normalize":
        norm = cm.colors.Normalize(col.min(), col.max())
    else: 
        raise ValueError("Invalid colormap scaling option. Choose from: Normalized")

    # Plot snapshot
    sc = ax.scatter(x, y, z,
                    c=col, norm=norm, cmap=cmap, s=10)
    # colorbar
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5)
    cbar.set_label(str(prop), rotation=90, labelpad=20)
    if prop == "v":
        cbar.set_label("logv", rotation=90, labelpad=20)
    
    return fig

def plot_vmax_fault(fault):
    """
    Plot time series of vmax for each fault
    """
    plt.rcParams.update(plt.rcParamsDefault)
    
    fig,ax= plt.subplots(ncols=1, nrows=2, figsize=figsize, squeeze=False)
    
    labels_fault = [str(i) for i in np.arange(1,len(fault)+1)]

    for i in np.arange(0,len(fault)):
        ax[i,0].plot(fault[i]["t"]/t_year, fault[i]["vmax_fault"])
        ax[i,0].set_xlabel("time (yr)")
        ax[i,0].set_ylabel("v (m/s)")
        ax[i,0].set_title("Fault "+labels_fault[i])
    
    fig.tight_layout()
    plt.show()
    return fig


def plot_frict_prop_1d(mesh_dict):
    """
    Plot 1D depth profile of A-B, A/B and SIGMA in the middle of the fault
    Works for a 2D fault
    If working with multiple faults, make sure to have filtered the mesh accordingly
    """


    # Find the middle point of x
    mid_pt = np.unique(mesh_dict['X'])[len(np.unique(mesh_dict['X'])) // 2]
    idx_mid = (mesh_dict['X']==mid_pt)

    # set arrays with quantities
    aminusb = mesh_dict["A"]-mesh_dict["B"]
    aob = mesh_dict["A"]/mesh_dict["B"]
    sigma = mesh_dict["SIGMA"]
    z = mesh_dict["Z"]

    fig, ax = plt.subplots(1, 3, sharey='row', figsize=[6, 6])
    fig.subplots_adjust(hspace=0.03)

    # A-B
    ax[0].grid(True)
    ax[0].set(title='A-B', xlim=[-0.01, 0.005], ylabel='Meters')
    for tick in ax[0].get_xticklabels():
        tick.set_rotation(40)
    ax[0].plot(aminusb[idx_mid],
            z[idx_mid],
            color='k', lw=1.5)

    #A/B
    ax[1].set(title='A / B', xlim=[0.3, 1.9])
    ax[1].grid(True)
    ax[1].plot(aob[idx_mid],
            z[idx_mid],
            color='k', lw=1.5)

    # sigma
    ax[2].set(title='Normal stress', xlabel='MPa')
    ax[2].grid(True)
    ax[2].plot(sigma[idx_mid],
            z[idx_mid],
            color='k')
    ax[2].set_xlim(-5e6, 60e6)

    fig.tight_layout()

    return fig

def plot_events(events_dict, fault_labels=[1]):
    """
    Uses the output from compute_events to plot a series of properties

    Parameters:
        events_dict - dictionary output from compute_events post-processing function
        fault_labels - array of labels of the faults to plot events for. Defaults to [1]

    Raises:
        ValueError - if fault_label not found in the input events dictionary

    Returns:
        None - returns matplotlib figure
    """
    # Set the rc parameters for marker size and linewidth
    plt.rc('lines', markersize=4, linewidth=0.5)

    # create canvas
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 7), squeeze=False)
    axs = ax.flatten()

    j=0 # for colour matching
    for i in fault_labels:
        if not i in events_dict["ev"]:
            raise ValueError(f"No fault with value {i} found in the events")

        df_evf = events_dict["ev"][i]

        #Slip
        axs[0].plot(df_evf["t_event"] / t_year, df_evf["cum_slip"], label = f"Fault {i}", marker='o')
        axs[0].set_xlabel("t [yr]")
        axs[0].set_ylabel("slip[m]")
        axs[0].legend()
        axs[0].set_title("Slip")

        # Peak slip rate
        axs[1].plot(df_evf["t_event"] / t_year, df_evf["peak_v"], label=f"Fault {i}", marker="o")
        axs[1].set_xlabel("t [yr]")
        axs[1].set_ylabel("slip rate [m/s]")
        axs[1].legend()
        axs[1].set_title("Peak slip rate")

        # Event duration
        axs[2].plot(df_evf["t_event"] / t_year, df_evf["dt_event"], label=f"Fault {i}", marker="o")
        axs[2].set_xlabel("t [yr]")
        axs[2].set_ylabel("t [s]")
        axs[2].legend()
        axs[2].set_title("Event duration")

        # Recurrence interval within fault
        axs[3].plot(df_evf["t_event"] / t_year, df_evf["t_interevent_intrafault"]/t_year, label=f"Fault {i}", marker="o")
        axs[3].set_xlabel("t [yr]")
        axs[3].set_ylabel("t [yr]")
        axs[3].legend()
        axs[3].set_title("Recurrence interval within fault")

        # Moment magnitude
        markerline, stemlines, baseline = axs[4].stem(df_evf["t_event"]/t_year,df_evf["Mw"],label = f"Fault {i}", markerfmt='o', basefmt= 'C0')
        plt.setp(markerline, color=f'C{j}')  # Set the marker color
        plt.setp(stemlines, color=f'C{j}')   # Set the stem color
        plt.setp(baseline, color=f'C{j}')    # Set the baseline color
        axs[4].set_ylim(bottom=0)
        axs[4].set_xlabel("t [yr]")
        axs[4].set_ylabel("Mw")
        axs[4].legend()
        axs[4].set_title("Moment magnitude")

        j += 1

    # Create a merged array from the separate faults
    df_rec = pd.concat(events_dict["ev"], keys=[i for i in fault_labels])

    # Resetting the index to make it unique
    df_rec.reset_index(level=0, drop=True, inplace=True)
    df_rec.index.name = 'n_event'

    # Sorting the combined dataframe by t_event
    df_rec.sort_values(by='t_event', inplace=True)

    # A new property for t_interevent_interfault
    df_rec['t_interevent_interfault'] = df_rec['t_event'].diff()

    axs[5].plot(df_rec["t_event"] / t_year, df_rec["t_interevent_interfault"]/t_year, marker='o')
    axs[5].set_xlabel("t [yr]")
    axs[5].set_ylabel("t [yr]")
    axs[5].set_title("Overall Recurrence Interval")



    fig.tight_layout()

    plt.show()


def cv_plot(events_dict, cv_keys = ['peak_v','dt_event','t_interevent_intrafault','Mw'], figsize=figsize):

    def coeff_vars(input):
        output_keys = cv_keys
        output = {}

        for i in range(1, input["mesh_dict"]["N_FAULTS"]+1):
            output[i] = {}

            for key in input["ev"][i].keys():
                if key in output_keys:
                    output[i][key] = (np.std(input["ev"][i][key]) / np.mean(input["ev"][i][key]))

        return output

    cv_data = coeff_vars(events_dict)

    metrics = list(next(iter(cv_data.values())).keys())  # ['peak_v', 'dt_event', 't_event', 't_interevent_intrafault', 'Mw']
    faults = sorted(cv_data.keys())
    num_metrics = len(metrics)

    # Preparing data for plotting
    x = np.arange(len(faults))  # label locations
    width = 0.15 # bar width

    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=figsize) # defaults to (9,8)

    # Plot bars for each metric
    for i, metric in enumerate(metrics):
        metric_values = [cv_data[fault][metric] for fault in faults]

        bar_positions = x + i * width

        ax.bar(bar_positions, metric_values, width, label=metric)

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_xlabel('Fault Number')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Coefficient of Variation for Different Metrics by Fault')
    ax.set_xticks(x + width * (num_metrics - 1) / 2)
    ax.set_xticklabels(faults)
    ax.legend(title='Metrics')

    # Display the plot
    plt.tight_layout()
    plt.show()

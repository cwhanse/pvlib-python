"""
Contains functions for solving for DC power in arrays with mismatched
conditions.

"""
import numpy as np
from scipy.optimize.elementwise import find_root
from pvlib import singlediode as _singlediode


def _iv_series_lambert_v_from_i(current, il, io, rs, rsh, a, neg_v_limit,
                                ndevices=None, idx=None):
    # wrapper for pvlib._singlediode._lambertw_v_from_i, handles
    # dimensions expected in series calculation
    # solve voltages at each current for each IV curve
    if ndevices is not None:
        # broadcast I to ndevices to apply same current each device
        current = np.broadcast_to(current[np.newaxis, :],
                                  (ndevices, len(current)))
    # slice each parameter on its ntimes dimension with idx
    if idx is not None:
        il, io, rs, rsh, a = (il[:, idx], io[:, idx], rs[:, idx], rsh[:, idx],
                              a[:, idx])

    voltages = _singlediode._lambertw_v_from_i(
        current.flatten(), il.flatten(), io.flatten(), rs.flatten(),
        rsh.flatten(), a.flatten())

    # apply negative voltage limit
    voltages[voltages < neg_v_limit] = neg_v_limit

    # reshape
    voltages = voltages.reshape(current.shape)
    return voltages


def _setup_currents(current_bkpts, string_isc, npts):
    r''' Form array of currents from string_isc down to 0.
    The array of currents will contain all values
    from current_bkpts which are less than string_isc.

    Parameters
    ----------
    device_isc : ndarray
        Shape (ndevices, ntimes)
    string_isc : ndarray
        Shape (ntimes,)
    npts : int
        number of current points in the returned array

    Returns
    -------
    ndarray
        shape (ntimes, npts)

    '''
    ntimes = len(string_isc)
    currents = np.zeros((ntimes, npts))

    u = current_bkpts < string_isc[np.newaxis, :]

    # have to loop on ntimes since count of device_isc < string_isc may
    # differ for each time
    for i in range(ntimes):
        vals = np.unique(current_bkpts[u[:, i], i])
        # ensure string_isc and 0. are added
        vals = np.append(vals, [string_isc[i], 0.])
        k_i = len(vals)

        if k_i == 0:
            continue

        # Copy original values
        currents[i, :k_i] = vals

        n_fill = npts - k_i
        if n_fill <= 0:
            continue

        # Build grid
        grid = np.linspace(string_isc[i], 0., npts)

        # Compute distance to nearest point in arr
        # shape: (grid_size, k_i)
        dists = np.abs(grid[:, None] - vals[None, :])

        # nearest distance per grid point
        nearest_dist = np.min(dists, axis=1)

        # inverse-distance weights (higher near arr values)
        weights = 1.0 / (nearest_dist + 1e-12)  # 1e-12 to avoid div by 0

        # Avoid re-selecting original values exactly
        mask_existing = np.isclose(nearest_dist, 0.0, atol=1e-12)
        weights[mask_existing] = 0.0

        # Select top-weighted grid points
        idx = np.argpartition(weights, -n_fill)[-n_fill:]
        selected = grid[idx]

        # Combine and add to A
        currents[i, :] = np.concatenate([vals, selected])

    # return sorted in descending order for each time
    currents = -np.sort(-currents, axis=1)

    return currents


def _iv_series_lambertw(photocurrent, saturation_current, resistance_series,
                        resistance_shunt, nNsVth, neg_v_limit=0.,
                        npts=100):
    r'''Solve the IV curve for series-connected devices where each device
    is described by the single diode equation.

    Uses a simplified model for reverse bias behavior, where current is
    unbounded at a constant reverse bias voltage ``neg_v_limit``.

    Input parameter ``photocurrent`` must have shape (devices, times).
    Input parameters ``saturation_current``, ``resistance_series``,
    ``resistance_shunt``, ``nNsVth`` may be arrays. If arrays, must be
    broadcastable to the shape of ``photocurrent``.

    Parameters
    ----------
    photocurrent : numeric
        photocurrent (A). Must have shape (devices, times).
    saturation_current : numeric
        saturation current (A). Must be broadcastable with photocurrent.
    resistance_series : numeric
        series resistance (ohm). Must be broadcastable with photocurrent.
    resistance_shunt : numeric
        shunt resistance (ohm). Must be broadcastable with photocurrent.
    nNsVth : numeric
        product of diode factor n, number of series cells Ns, and
        thermal voltage (Vth), (V). Must be broadcastable with photocurrent.
    neg_v_limit : float, optional
        Limit on reverse bias voltage, from cell breakdown voltage or reverse
        bias diode activation voltage (V). Should be negative. For example,
        if neg_v_limit=-5, then at V=-5 current is unbounded in the positive
        direction.
    npts : int, optional
        Number of points used to discretize the returned IV curves.

    Returns
    -------
    voltages : numeric
        Voltage points for the series IV curves (V), shape
        (times, npts).
    currents : numeric
        Current points for the series IV curves (A), shape
        (times, npts).

    '''
    # target shape is ndevices x ntimes
    IL, I0, Rs, Rsh, a = \
        np.broadcast_arrays(photocurrent, saturation_current,
                            resistance_series, resistance_shunt, nNsVth)

    ndevices, ntimes = IL.shape

    # solve for current at negative voltage limit for each device.
    # these currents create breakpoints in the series IV curve
    current_bkpts = _singlediode._lambertw_i_from_v(
        neg_v_limit, IL, I0, Rs, Rsh, a)

    # find Isc for string IV curve
    # bounds, 1d array for each time
    max_isc = current_bkpts.max(axis=0) * 1.01
    min_isc = current_bkpts.min(axis=0) * 0.99

    # Use an index idx so that find_root can slice arguments
    # Internally find_root will slice current as each element converges
    # As an argument, idx lets find_root also slice the other parameters
    # Remove idx and use preserve_shape once available in find_root
    # https://github.com/scipy/scipy/issues/24869
    idx = np.arange(ntimes)

    def optfn(current, idx):
        # current is ntimes only since it is common for all devices.
        # other parameters are ntimes x ndevices
        v = _iv_series_lambert_v_from_i(
            current, IL, I0, Rs,
            Rsh, a, neg_v_limit, ndevices, idx)
        # return string voltage
        return v.sum(axis=0)

    isc_result = find_root(
        optfn,
        (min_isc, max_isc), args=(idx,))
    string_isc = isc_result.x  # 1d in ntimes

    # discretize current from string_isc down to 0 at each time step
    # Include each device Isc (except the highest) so that the series IV curve
    # includes the breakpoints
    current_pts = _setup_currents(current_bkpts, string_isc, npts)

    # shape all arrays to be ndevices x ntimes x ncurrents
    curs = np.repeat(current_pts[np.newaxis, :, :], ndevices, axis=0)
    curs, IL, I0, Rs, Rsh, a = np.broadcast_arrays(
        curs, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    voltages = _iv_series_lambert_v_from_i(
        curs, IL, I0, Rs, Rsh, a, neg_v_limit)

    # add voltage across devices to get series voltage
    voltage_sum = voltages.sum(axis=0)

    # drop currents dimension for devices
    curs = curs[0, :, :]

    return voltage_sum, curs

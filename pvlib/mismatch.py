"""
Contains functions for solving for DC power in arrays with mismatched
conditions.

"""
import numpy as np
from scipy.optimize.elementwise import find_root
import singlediode as _singlediode


def _iv_series_lambert_v_from_i(I, il, io, rs, rsh, a, neg_v_limit,
                                ndevices=None, idx=None):
    # solve voltages at each current for each IV curve
    if ndevices is not None:
        # broadcast I to ndevices to apply same current each device
        I = np.broadcast_to(I[np.newaxis, :], (ndevices, len(I)))
    # slice each parameter on its ntimes dimension with idx
    if idx is not None:
        il, io, rs, rsh, a = (il[:, idx], io[:, idx], rs[:, idx], rsh[:, idx],
                              a[:, idx])

    voltages = _singlediode._lambertw_v_from_i(
        I.flatten(), il.flatten(), io.flatten(), rs.flatten(), rsh.flatten(),
        a.flatten())

    # apply negative voltage limit
    voltages[voltages < neg_v_limit] = neg_v_limit

    # reshape
    voltages = voltages.reshape(I.shape)
    return voltages


def _setup_currents(device_isc, string_isc, npts):
    r''' Form array of currents. Array of currents will contain all values
    from device_isc which are less than string_isc.

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
    A = np.zeros((ntimes, npts))

    u = device_isc < string_isc[np.newaxis, :]

    # have to loop on ntimes since count of device_isc < string_isc may
    # differ for each time
    for i in range(ntimes):
        vals = np.unique(device_isc[u[:, i], i])
        # ensure string_isc and 0. are added
        vals = np.append(vals, [string_isc[i], 0.])
        k_i = len(vals)

        if k_i == 0:
            continue

        # Copy original values
        A[i, :k_i] = vals

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
        A[i, :] = np.concatenate([vals, selected])

    # return sorted in descending order for each time
    A = -np.sort(-A, axis=1)

    return A


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

    # solve for Isc
    # Isc for each device
    device_isc = _singlediode._lambertw_i_from_v(
        neg_v_limit, IL, I0, Rs, Rsh, a)

    # find Isc for series of devices
    # 1d array contains bounds for each time
    # add/substract eps in case max==min
    max_isc = device_isc.max(axis=0) * 1.01
    min_isc = device_isc.min(axis=0) * 0.99

    # use index idx so that find_root can slice arguments
    # internally find_root will slice current as each element converges
    # as an argument, idx lets find_root also slice the other parameters
    # remove idx and use preserve_shape once available in find_root
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

    # discretize current from max(Isc) down to 0. at each time step
    # Include each device Isc (except the highest) so that the series IV curve
    current_pts = _setup_currents(device_isc, string_isc, npts)

    # shape all arrays to be ndevices x ntimes x ncurrents
    I = np.repeat(current_pts[np.newaxis, :, :], ndevices, axis=0)
    I, IL, I0, Rs, Rsh, a = np.broadcast_arrays(
        I, IL[:, :, np.newaxis], I0[:, :, np.newaxis], Rs[:, :, np.newaxis],
        Rsh[:, :, np.newaxis], a[:, :, np.newaxis])

    # solve voltages at each current for each IV curve
    voltages = _iv_series_lambert_v_from_i(
        I, IL, I0, Rs, Rsh, a, neg_v_limit)

    # add voltage across devices to get series voltage
    voltage_sum = voltages.sum(axis=0)

    # drop currents dimension for devices
    I = I[0, :, :]

    return voltage_sum, I

import importlib
import time
import os

import torch
import numpy as np

import math
import librosa
import astropy.coordinates as coord
import astropy.units as u
import scipy.constants as constants
import scipy.linalg as linalg
import scipy.signal.windows as windows
import skimage.util as skutil


def load_checkpoint(checkpoint_path, device):
    _, ext = os.path.splitext(os.path.basename(checkpoint_path))
    assert ext in (".pth", ".tar"), "Only support ext and tar extensions of model checkpoint."
    model_checkpoint = torch.load(checkpoint_path, map_location=device)

    if ext == ".pth":
        print(f"Loading {checkpoint_path}.")
        return model_checkpoint
    else:  # tar
        print(f"Loading {checkpoint_path}, epoch = {model_checkpoint['epoch']}.")
        return model_checkpoint["model"]


def prepare_empty_dir(dirs, resume=False):
    """
    if resume experiment, assert the dirs exist,
    if not resume experiment, make dirs.

    Args:
        dirs (list): directors list
        resume (bool): whether to resume experiment, default is False
    """
    for dir_path in dirs:
        if resume:
            assert dir_path.exists()
        else:
            dir_path.mkdir(parents=True, exist_ok=True)


class ExecutionTime:
    """
    Usage:
        timer = ExecutionTime()
        <Something...>
        print(f'Finished in {timer.duration()} seconds.')
    """

    def __init__(self):
        self.start_time = time.time()

    def duration(self):
        return int(time.time() - self.start_time)


def initialize_config(module_cfg, pass_args=True):
    """According to config items, load specific module dynamically with params.
    e.g., Config items as follow：
        module_cfg = {
            "module": "model.model",
            "main": "Model",
            "args": {...}
        }
    1. Load the module corresponding to the "module" param.
    2. Call function (or instantiate class) corresponding to the "main" param.
    3. Send the param (in "args") into the function (or class) when calling ( or instantiating)
    """
    module = importlib.import_module(module_cfg["module"])

    if pass_args:
        return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else:
        return getattr(module, module_cfg["main"])


def z_score(m):
    mean = np.mean(m)
    std_var = np.std(m)
    return (m - mean) / std_var, mean, std_var


def reverse_z_score(m, mean, std_var):
    return m * std_var + mean


def min_max(m):
    m_max = np.max(m)
    m_min = np.min(m)

    return (m - m_min) / (m_max - m_min), m_max, m_min


def reverse_min_max(m, m_max, m_min):
    return m * (m_max - m_min) + m_min


def sample_fixed_length_data_aligned(data_a, data_b, sample_length):
    """Sample with fixed length from two datasets, with zero padding if needed."""
    frames_total = data_a.shape[1]
    # Pad with zeros if the total frames are less than the sample length
    if frames_total < sample_length:
        padding_length = sample_length - frames_total
        data_a = np.pad(data_a, ((0, 0), (0, padding_length)), mode='constant')
        data_b = np.pad(data_b, ((0, 0), (0, padding_length)), mode='constant')
        start = 0  # Start at the beginning since padding is added to meet the length
    else:
        start = np.random.randint(frames_total - sample_length + 1)

    end = start + sample_length
    return data_a[:, start:end], data_b[:, start:end]


def print_tensor_info(tensor, flag="Tensor"):
    floor_tensor = lambda float_tensor: int(float(float_tensor) * 1000) / 1000
    print(flag)
    print(
        f"\tmax: {floor_tensor(torch.max(tensor))}, min: {float(torch.min(tensor))}, mean: {floor_tensor(torch.mean(tensor))}, std: {floor_tensor(torch.std(tensor))}")



############################################################################
###################### Phased Array Utility Functions ######################
############################################################################

__eigenmike_raw__ = {
    "1": [69, 0, 0.042], "2": [90, 32, 0.042], "3": [111, 0, 0.042], "4": [90, 328, 0.042], "5": [32, 0, 0.042],
    "6": [55, 45, 0.042], "7": [90, 69, 0.042], "8": [125, 45, 0.042], "9": [148, 0, 0.042], "10": [125, 315, 0.042],
    "11": [90, 291, 0.042], "12": [55, 315, 0.042], "13": [21, 91, 0.042], "14": [58, 90, 0.042], "15": [121, 90, 0.042],
    "16": [159, 89, 0.042], "17": [69, 180, 0.042], "18": [90, 212, 0.042], "19": [111, 180, 0.042], "20": [90, 148, 0.042],
    "21": [32, 180, 0.042], "22": [55, 225, 0.042], "23": [90, 249, 0.042], "24": [125, 225, 0.042],"25": [148, 180, 0.042],
    "26": [125, 135, 0.042], "27": [90, 111, 0.042], "28": [55, 135, 0.042], "29": [21, 269, 0.042], "30": [58, 270, 0.042],
    "31": [122, 270, 0.042], "32": [159, 271, 0.042],
}

def _deg2rad(coords_dict):
    """
    Take a dictionary with microphone array
    capsules and 3D polar coordinates to
    convert them from degrees to radians
    colatitude, azimuth, and radius (radius
    is left intact)
    """
    return {
        m: [math.radians(c[0]), math.radians(c[1]), c[2]]
        for m, c in coords_dict.items()
    }


def _polar2cart(coords_dict, units=None):
    """
    Take a dictionary with microphone array
    capsules and polar coordinates and convert
    to cartesian
    Parameters:
        units: (str) indicating 'degrees' or 'radians'
    """
    if units == None or units != "degrees" and units != "radians":
        raise ValueError("you must specify units of 'degrees' or 'radians'")
    elif units == "degrees":
        coords_dict = _deg2rad(coords_dict)
    return {
        m: [
            c[2] * math.sin(c[0]) * math.cos(c[1]),
            c[2] * math.sin(c[0]) * math.sin(c[1]),
            c[2] * math.cos(c[0]),
        ]
        for m, c in coords_dict.items()
    }

def eq2cart(r, lat, lon):
    r = np.array([r]) #if chk.is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
        .to_cartesian()
        .xyz.to_value(u.dimensionless_unscaled)
    )
    return XYZ


def pol2cart(r, colat, lon):
    lat = (np.pi / 2) - colat
    return eq2cart(r, lat, lon)


def fibonacci(N, direction=None, FoV=None, shift_lon=0, shift_colat=0):            
    if direction is not None:
        direction = np.array(direction, dtype=float)
        direction /= linalg.norm(direction)

        if FoV is not None:
            if not (0 < np.rad2deg(FoV) < 360):
                raise ValueError("Parameter[FoV] must be in (0, 360) degrees.")
        else:
            raise ValueError("Parameter[FoV] must be specified if Parameter[direction] provided.")

    if N < 0:
        raise ValueError("Parameter[N] must be non-negative.")

    N_px = 4 * (N + 1) ** 2
    n = np.arange(N_px)

    colat = np.arccos(1 - (2 * n + 1) / N_px)
    lon = (4 * np.pi * n) / (1 + np.sqrt(5)) + shift_lon
    XYZ = np.stack(pol2cart(1, colat, lon), axis=0)

    if direction is not None:  # region-limited case.
        # TODO: highly inefficient to generate the grid this way!
        min_similarity = np.cos(FoV / 2)
        mask = (direction @ XYZ) >= min_similarity
        XYZ = XYZ[:, mask]

    return XYZ


def get_field(min_freq=1500, max_freq=4500,nbands=10, shift_lon=0, shift_colat=0):
    """
    Get the field of view for the microphone array.
    """
    freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(min_freq, max_freq, 10), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]

    xyz = get_xyz()
    dev_xyz = np.array(xyz).T
    wl_min = constants.speed_of_sound / (freq.max() + 500)
    #sh_order = nyquist_rate(dev_xyz, wl_min) # Maximum order of complex plane waves that can be imaged by the instrument.
    sh_order=10
    R = fibonacci(sh_order, direction=None, FoV=None, shift_lon=shift_lon, shift_colat=shift_colat)
    R_mask = np.abs(R[2, :]) < np.sin(np.deg2rad(90))  # Visible region mask.
    R = R[:, R_mask]  # Shrink visible view to avoid border effects.
    return R


def get_xyz():
    """
    Get Cartesian coordinates of microphone array capsules.
    """
    mic_coords = _polar2cart(__eigenmike_raw__, units='degrees')
    xyz = [[coord for coord in mic_coords[ch]] for ch in mic_coords]
    return xyz


def steering_operator():
    r"""
    Steering matrix.

    Parameters
    ----------
    XYZ : :py:class:`~numpy.ndarray`
        (3, N_antenna) Cartesian array geometry.
    R : :py:class:`~numpy.ndarray`
        (3, N_px) Cartesian grid points in :math:`\mathbb{S}^{2}`.

    Returns
    -------
    A : :py:class:`~numpy.ndarray`
        (N_antenna, N_px) steering matrix.

    Notes
    -----
    The steering matrix is defined as:

    .. math:: {\bf{A}} = \exp \left( -j \frac{2 \pi}{\lambda} {\bf{P}}^{T} {\bf{R}} \right),

    where :math:`{\bf{P}} \in \mathbb{R}^{3 \times N_{\text{antenna}}}` and
    :math:`{\bf{R}} \in \mathbb{R}^{3 \times N_{\text{px}}}`.
    """
    xyz = get_xyz()
    XYZ = np.array(xyz).T
    R = get_field()
    freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(1500, 4500, 10), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]
    wl = constants.speed_of_sound / (freq.max() + 500)
    if wl <= 0:
        raise ValueError("Parameter[wl] must be positive.")

    scale = 2 * np.pi / wl
    A = np.exp((-1j * scale * XYZ.T) @ R)
    return A



def extract_visibilities(_data, _rate, T, fc, bw, alpha):
    """
    Transform time-series to visibility matrices.

    Parameters
    ----------
    T : float
        Integration time [s].
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    alpha : float
        Shape parameter of the Tukey window, representing the fraction of
        the window inside the cosine tapered region. If zero, the Tukey
        window is equivalent to a rectangular window. If one, the Tukey
        window is equivalent to a Hann window.

    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices (complex-valued).
    """
    N_stft_sample = int(_rate * T)
    if N_stft_sample == 0:
        raise ValueError('Not enough samples per time frame.')
    # print(f'Samples per STFT: {N_stft_sample}')

    N_sample = (_data.shape[0] // N_stft_sample) * N_stft_sample
    N_channel = _data.shape[1]
    stf_data = (skutil.view_as_blocks(_data[:N_sample], (N_stft_sample, N_channel))
                .squeeze(axis=1))  # (N_stf, N_stft_sample, N_channel)

    window = windows.tukey(M=N_stft_sample, alpha=alpha, sym=True).reshape(1, -1, 1)
    stf_win_data = stf_data * window  # (N_stf, N_stft_sample, N_channel)
    N_stf = stf_win_data.shape[0]

    stft_data = np.fft.fft(stf_win_data, axis=1)  # (N_stf, N_stft_sample, N_channel)
    # Find frequency channels to average together.
    idx_start = int((fc - 0.5 * bw) * N_stft_sample / _rate)
    idx_end = int((fc + 0.5 * bw) * N_stft_sample / _rate)
    collapsed_spectrum = np.sum(stft_data[:, idx_start:idx_end + 1, :], axis=1)

    # Don't understand yet why conj() on first term?
    # collapsed_spectrum = collapsed_spectrum[0,:]
    S = (collapsed_spectrum.reshape(N_stf, -1, 1).conj() *
        collapsed_spectrum.reshape(N_stf, 1, -1))
    return S

def form_visibility(data, rate, fc, bw, T_sti, T_stationarity):
    '''
    Parameter
    ---------
    data : :py:class:`~numpy.ndarray`
        (N_sample, N_channel) antenna samples. (float)
    rate : int
        Sample rate [Hz]
    fc : float
        Center frequency [Hz] around which visibility matrices are formed.
    bw : float
        Double-wide bandwidth [Hz] of the visibility matrix.
    T_sti : float
        Integration time [s]. (time-series)
    T_stationarity : float
        Integration time [s]. (visibility)
        
    Returns
    -------
    S : :py:class:`~numpy.ndarray`
        (N_slot, N_channel, N_channel) visibility matrices.
        
        # N_slot == number of audio frames in track

    Note
    ----
    Visibilities computed directly in the frequency domain.
    For some reason visibilities are computed correctly using
    `x.reshape(-1, 1).conj() @ x.reshape(1, -1)` and not the converse.
    Don't know why at the moment.
    '''
    S_sti = (extract_visibilities(data, rate, T_sti, fc, bw, alpha=1.0))

    N_sample, N_channel = data.shape
    N_sti_per_stationary_block = int(T_stationarity / T_sti)
    S = (skutil.view_as_windows(S_sti,
                                (N_sti_per_stationary_block, N_channel, N_channel),
                                (N_sti_per_stationary_block, N_channel, N_channel))
        .squeeze(axis=(1, 2))
        .sum(axis=1))
    return S

def get_visibility_matrix(audio_in, fs, T_sti=10e-3, scale="linear", nbands=9):
    freq, bw = None, None
    if scale == "linear":
        freq, bw = (skutil  # Center frequencies to form images
            .view_as_windows(np.linspace(1500, 4500, nbands+1), (2,), 1)
            .mean(axis=-1)), 50.0  # [Hz]
    elif scale == "log":
        freq, bw = librosa.mel_frequencies(n_mels=nbands, fmin=50, fmax=4500), 50
        print(f"Log scale frequencies: {freq}, bandwidth: {bw}")
    else:
        raise Exception("Not a valid scale to generate covariance matrices (log, linear)")
    
    visibilities = []

    for i in range(nbands):
        T_stationarity = 10 * T_sti  # Choose to have frame_rate = 10
        S = form_visibility(audio_in, fs, freq[i], bw, T_sti, T_stationarity)
        N_sample = S.shape[0]
        visibilities_per_frame = []

        for s_idx in range(N_sample):
            S_D, S_V = linalg.eigh(S[s_idx])
            if S_D.max() <= 0:
                S_D[:] = 0
            else:
                S_D = np.clip(S_D / S_D.max(), 0, None)
            S_norm = (S_V * S_D) @ S_V.conj().T
            # S_norm = S[s_idx]   
            visibilities_per_frame.append(S_norm)

        visibilities.append(visibilities_per_frame)

    return np.array(visibilities)

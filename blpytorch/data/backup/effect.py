import numpy as np
import scipy.signal as s
import torch
import torchaudio.transforms as T
from . import helper as h

from ..modules.fDomainHelper import FDomainHelper
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import sosfiltfilt
from scipy.signal import resample_poly

EPS = 1e-8


def reverb_rir(frames, rir):
    """
    frames is the clean audio numpy with shape [T]
    rir is the rir numpy with shape [T']
    they should share the same sample rate (44100 hz)
    returns: reverberated audio with shape [T'] (numpy)
    """
    orig_frames_shape = frames.shape
    frames, filter = np.squeeze(frames), np.squeeze(rir)
    frames = s.convolve(frames, filter)
    actlev = np.max(np.abs(frames))
    if actlev > 0.99:
        frames = (frames / actlev) * 0.98
    frames = frames[: orig_frames_shape[0]]
    # print(frames.shape, orig_frames_shape)
    return frames


def clip(frames, f):
    """
    f is value between [0,1)
    frames: 1d numpy audio array
    """
    min_value = np.min(frames)
    max_value = np.max(frames)
    return np.clip(frames, min_value * f, max_value * f)


def add_noise_and_scale(HQ, augfront, noise, snr, scale):
    """
    :param HQ: the clean audio without any distortions and noise
    :param augfront: front-head audio, like vocal [samples,channel], will be normlized so any scale will be fine
    :param noise: noise, [samples,channel], any scale
    :param snr: snr ratio
    :param scale: scaling factor
    :return: scaled front and noise (noisy = front + noise), all_mel_e2e outputs are noramlized within [-1 , 1]
    """
    noise = _normalize_energy(noise)  # set noise and vocal to equal range [-1,1]
    HQ, augfront = _unify_energy(HQ, augfront)
    # some clipping noise is extremly noisy
    front_level = np.mean(np.abs(augfront))
    if front_level > 0.02:
        noise_front_energy_ratio = np.mean(np.abs(noise)) / front_level
        noise = noise / noise_front_energy_ratio
    noise = _norm_noise(noise, snr)
    noisy, augfront, noise, HQ = _unify_energy(
        noise + augfront, augfront, noise, HQ
    )  # normalize noisy, noise and vocal energy into [-1,1]
    # print("unify:", np.max(noise), np.max(noisy), np.max(HQ), np.max(augfront))
    noise, augfront, HQ = noise * scale, augfront * scale, HQ * scale  # apply scale
    # print("after scale", torch.max(noisy), torch.max(noise), torch.max(front), snr, scale)
    # print("snr:", SNR(augfront, noise))
    return HQ, noisy


def add_noise_and_scale_with_HQ_with_Aug(HQ, front, augfront, noise, snr, scale):
    """
    :param front: front-head audio, like vocal [samples,channel], will be normlized so any scale will be fine
    :param noise: noise, [samples,channel], any scale
    :param snr_l: Optional
    :param snr_h: Optional
    :param scale_lower: Optional
    :param scale_upper: Optional
    :return: scaled front and noise (noisy = front + noise), all_mel_e2e outputs are noramlized within [-1 , 1]
    """
    noise = _normalize_energy(noise)  # set noise and vocal to equal range [-1,1]
    HQ, front, augfront = _unify_energy(HQ, front, augfront)
    # some clipping noise is extremly noisy
    front_level = np.mean(np.abs(augfront))
    if front_level > 0.02:
        noise_front_energy_ratio = np.mean(np.abs(noise)) / front_level
        noise = noise / noise_front_energy_ratio
    noise = _norm_noise(noise, snr)
    _, augfront, noise, front, HQ = _unify_energy(
        noise + augfront, augfront, noise, front, HQ
    )  # normalize noisy, noise and vocal energy into [-1,1]
    # print("unify:", torch.max(noise), torch.max(front), torch.max(noisy))
    # print("Scale",scale)
    noise, front, augfront, HQ = (
        noise * scale,
        front * scale,
        augfront * scale,
        HQ * scale,
    )  # apply scale
    # print("after scale", torch.max(noisy), torch.max(noise), torch.max(front), snr, scale)
    # print("snr:", SNR(HQ, noise))
    return HQ, noise + augfront


def signalPower(x):
    return np.average(x**2)


def SNR(signal, noise):
    powS = signalPower(signal)
    powN = signalPower(noise)
    return 10 * np.log10((powS) / powN)


def _activelev(*args):
    """
    need to update like matlab
    """
    res = torch.concat(list(args))
    return torch.max(torch.abs(res))


def _norm_noise(noise, snr):
    """
    noise = noise / (10^(s/20))
    """
    clean_weight = 10 ** (float(snr) / 20)
    return noise / clean_weight


def _normalize_energy(audio, alpha=1):
    """
    :param audio: 1d waveform, [batchsize, *],
    :param alpha: the value of output range from: [-alpha,alpha]
    :return: 1d waveform which value range from: [-alpha,alpha]
    """
    val_max = _activelev(audio)
    if val_max == 0:
        return audio
    return (audio / val_max) * alpha


def _unify_energy(*args):
    max_amp = _activelev(*args)
    mix_scale = 1.0 / max_amp
    return [x * mix_scale for x in args]


def _clip_and_rir(frames: np.number, config: dict, rir_path, fs):
    _frames = frames.copy()
    ### clip
    res = {}
    is_clip: bool = h.uniform_sample(prob=config["clip"]["prob"])
    if is_clip:
        ## doing the clipping thing:
        low, high = config["clip"]["range"][0], config["clip"]["range"][1]
        clip_factor = h.uniform_sample(lower=low, upper=high)
        _frames = clip(_frames, clip_factor)
        res["clip"] = clip_factor
    ### rir
    is_rir: bool = h.uniform_sample(prob=config["rir"]["prob"])
    if is_rir:
        try:
            rir = h.random_choice(rir_path)
            rir_audio = h.readwav(rir, fs)
        except ValueError:
            rir = "/public/home/qinxy/bltang/data/DNS-Challenge/dns_2022/datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k/largeroom/Room174/Room174-00093.wav"
            rir_audio = h.readwav(rir, fs)
            ### convolve the rir with the audio
        _frames = reverb_rir(frames=_frames, rir=rir_audio)
        res["rir"] = rir
    return _frames, res  # [vocal_aug]


def _apply_low_pass(_frames, chance, cutoff, fs, order, filter, mode="vocal"):
    frames = _frames.copy()
    if mode == "vocal":
        frames = lowpass(data=frames, highcut=cutoff, fs=fs, order=order, _type=filter)
        if _get_chance(chance):
            frames = lowpass(
                data=frames, highcut=cutoff, fs=fs, order=order, _type="stft"
            )
    elif mode == "noise":
        if _get_chance(chance):
            pass
        else:
            frames = lowpass(
                data=frames, highcut=cutoff, fs=fs, order=order, _type=filter
            )
            if _get_chance(chance, 3):
                frames = lowpass(
                    data=frames, highcut=cutoff, fs=fs, order=order, _type="stft"
                )
    else:
        raise ValueError("the mode is wrong, it should be either vocal or noise")
    return frames


def augment(
    frames: np.number, config: dict, rir_path, fs, mode="vocal", low_res=True
) -> np.number:
    """
    type can be either vocal or noise,
    rir_path: list of rir
    low_res: True if want to add low bandwitdh / False if not added
    return vocal, vocal_LR, vocal_aug_LR if mode == vocal
    return noise_LR if mode == noise
    """
    ori_len = frames.shape[-1]
    result = {}
    if mode == "vocal":
        frames_aug, res = _clip_and_rir(frames, config, rir_path, fs)
        result.update(res)
    elif mode == "noise":
        frames_aug = frames.copy()
    else:
        raise ValueError(f"mode has to be vocal or noise")
    ### low resolution
    is_low_pass = h.uniform_sample(prob=config["low_pass"]["prob"])
    if not low_res or not is_low_pass:
        if mode == "vocal":
            return (
                frames,
                frames.copy(),
                frames_aug.copy(),
                result,
            )  # [vocal , vocal_LR, vocal_aug_LR]
        elif mode == "noise":
            return frames_aug, result  # [noise_LR]
        else:
            raise ValueError("mode has to be vocal or noise")
    cutoff = h.uniform_sample(
        lower=config["low_pass"]["range"][0] // 2,
        upper=config["low_pass"]["range"][1] // 2,
        return_int=True,
    )
    order = h.uniform_sample(range=config["low_pass"]["order"], return_int=True)
    filter = h.random_choice(config["low_pass"]["type"])
    chance = h.uniform_sample(lower=0, upper=1000, return_int=True)
    (
        result[f"order_{mode}"],
        result[f"filter_{mode}"],
        result[f"chance_{mode}"],
        result[f"cutoff_{mode}"],
    ) = (
        order,
        filter,
        chance,
        cutoff,
    )
    if mode == "vocal":
        frames_LR = _apply_low_pass(frames, chance, cutoff, fs, order, filter)
        frames_aug_LR = _apply_low_pass(frames_aug, chance, cutoff, fs, order, filter)
        frames = h.constrain_length(frames, ori_len)
        frames_LR = h.constrain_length(frames_LR, ori_len)
        frames_aug_LR = h.constrain_length(frames_aug_LR, ori_len)
        return frames, frames_LR, frames_aug_LR, result
    elif mode == "noise":
        frames_LR = _apply_low_pass(frames, chance, cutoff, fs, order, filter)
        frames_LR = h.constrain_length(frames_LR, ori_len)
        return frames_LR, result


def _get_chance(chance, mode=2):
    # return False
    return int(chance) % mode == 0


class LowPass:
    f_helper = FDomainHelper()

    def __init__(self, cutoff: list, order: list, filter_type: list, fs=44100):
        ## sample from distribution
        self.cutoff = h.uniform_sample(
            lower=cutoff[0], upper=cutoff[1], return_int=True
        )
        self.order = h.uniform_sample(lower=order[0], upper=order[1], return_int=True)
        self.filter = h.random_choice(filter_type)
        self.fs = fs

    def low_resolution(self, frames):
        """frames: 1d numpy audio"""
        cutoff = self.cutoff
        order = self.order
        filter = self.filter
        result = LowPass._lowpass_filter(frames, cutoff, self.fs, order, filter)
        result = LowPass.resample_audio(result, cutoff * 2, self.fs)
        return result

    @staticmethod
    def resample_audio(audio, cutoff, original):
        """audio 1D"""
        y = resample_poly(audio, cutoff, original)
        y = resample_poly(y, original, cutoff)
        if len(y) != len(audio):
            y = LowPass._align_length(audio, y)
        return y

    @staticmethod
    def _align_length(x, y):
        """align the length of y to that of x, 1 D

        Args:
            x (np.array): reference signal
            y (np.array): the signal needs to be length aligned

        Return:
            yy (np.array): signal with the same length as x
        """
        Lx = len(x)
        Ly = len(y)

        if Lx == Ly:
            return y
        elif Lx > Ly:
            # pad y with zeros
            return np.pad(y, (0, Lx - Ly), mode="constant")
        else:
            # cut y
            return y[:Lx]

    @staticmethod
    def _lowpass_filter(x, highcut, fs, order, ftype):
        """process input signal x using lowpass filter

        Args:
            x (np.array): input signal
            highcut (float): high cutoff frequency
            order (int): the order of filter
            ftype (string): type of filter
                ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

        Return:
            y (np.array): filtered signal
        """
        nyq = 0.5 * fs
        hi = highcut / nyq
        if ftype == "butter":
            sos = butter(order, hi, btype="low", output="sos")
        elif ftype == "cheby1":
            sos = cheby1(order, 0.1, hi, btype="low", output="sos")
        elif ftype == "cheby2":
            sos = cheby2(order, 60, hi, btype="low", output="sos")
        elif ftype == "ellip":
            sos = ellip(order, 0.1, 60, hi, btype="low", output="sos")
        elif ftype == "bessel":
            sos = bessel(order, hi, btype="low", output="sos")
        else:
            raise Exception(f"The lowpass filter {ftype} is not supported!")
        y = sosfiltfilt(sos, x)

        if len(y) != len(x):
            y = LowPass._align_length(x, y)
        return y

    pass


def lowpass(data, highcut, fs, order=5, _type="butter"):
    """
    :param data: np.float32 type 1d time numpy array, (samples,) , can not be (samples, 1) !!!!!!!!!!!!
    :param highcut: cutoff frequency
    :param fs: sample rate of the original data
    :param order: order of the filter
    :return: filtered data, (samples,)
    """

    if len(list(data.shape)) != 1:
        raise ValueError(
            "Error (chebyshev_lowpass_filter): Data "
            + str(data.shape)
            + " should be type 1d time array, (samples,) , can not be (samples, 1)"
        )
    if _type in "butter":
        order = limit(order, high=10, low=2)
        return LowPass._lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="butter"
        )
    elif _type in "cheby1":
        order = limit(order, high=10, low=2)
        return LowPass._lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="cheby1"
        )
    # elif(_type in "cheby2"):
    #     order = limit(order, high=10, low=2)
    #     return lowpass_filter(x=data, highcut=int(highcut), fs=fs, order=order, ftype="cheby2")
    elif _type in "ellip":
        order = limit(order, high=10, low=2)
        return LowPass._lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="ellip"
        )
    elif _type in "bessel":
        order = limit(order, high=10, low=2)
        return LowPass._lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="bessel"
        )
    elif _type in "stft":
        return stft_hard_lowpass(data, lowpass_ratio=highcut / int(fs / 2))
    elif _type in "stft_hard":
        return stft_hard_lowpass_v0(data, lowpass_ratio=highcut / int(fs / 2))
    else:
        raise ValueError("Error: Unexpected filter type " + _type)


def limit(integer, high, low):
    if integer > high:
        return high
    elif integer < low:
        return low
    else:
        return int(integer)


def stft_hard_lowpass(data, lowpass_ratio, fs_ori=44100):
    fs_down = int(lowpass_ratio * fs_ori)
    # downsample to the low sampling rate
    y = resample_poly(data, fs_down, fs_ori)

    # upsample to the original sampling rate
    y = resample_poly(y, fs_ori, fs_down)

    if len(y) != len(data):
        y = LowPass._align_length(data, y)
    return y


def stft_hard_lowpass_v0(data, lowpass_ratio):
    length = data.shape[0]
    if type(data) is not torch.Tensor:
        data = torch.tensor(data, dtype=torch.float32)
    else:
        data = data.float()
    sps, coss, sins = LowPass.f_helper.wav_to_spectrogram_phase(data[None, None, ...])
    cut_frequency_dim = int(sps.size()[-1] * lowpass_ratio)
    sps[..., cut_frequency_dim:] = torch.zeros_like(sps[..., cut_frequency_dim:])
    data = LowPass.f_helper.spectrogram_phase_to_wav(sps, coss, sins, length)
    data = data[0, 0, :].numpy()
    return data

import torch
import random
import torch.nn.functional as F


def truc_wav(*x: torch.Tensor, length=None):
    """
    Given a list of tensors with the same length as arguments, chunk the tensors into a given length.
    Note that all the tensors will be chunked using the same offset
    x: [T] or [T,*]
    Args:
        x: the list of tensors to be chunked, should have the same length with shape [T, E]
        length: the length to be chunked into, if length is None, return the original audio
    Returns:
        A list of chuncked tensors
    """
    x_len = x[0].size(0)  # [T]
    res = []
    if length == None:
        for a in x:
            res.append(a)
        return res[0] if len(res) == 1 else res
    if x_len > length:
        offset = random.randint(0, x_len - length - 1)
        for a in x:
            res.append(a[offset : offset + length])
    # else:
    #     for a in x:
    #         res.append(F.pad(a, (0, 0, 0, length - a.size(0)), "constant"))
    else:
        for a in x:
            padding = [0] * (a.dim() - 1) * 2 + [
                0,
                length - a.size(0),
            ]  # Only pad the first dimension
            res.append(F.pad(a, padding, "constant"))
    return res[0] if len(res) == 1 else res


def _activelev(*args):
    """
    need to update like matlab
    """
    res = torch.concat(list(args))
    return torch.max(torch.abs(res))


def unify_energy(*args):
    max_amp = _activelev(*args)
    mix_scale = 1.0 / max_amp
    return [x * mix_scale for x in args]


def generate_target_audio(spk1, spk2, regi, mix_length, regi_length, snr=5):
    """
    spk 1: T1
    spk 2: T2
    regi: T3
    """
    spk1, spk2 = unify_energy(spk1, spk2)
    snr_1 = random.random() * snr / 2
    snr_2 = -snr_1
    spk1 = spk1 * 10 ** (snr_1 / 20)
    spk2 = spk2 * 10 ** (snr_2 / 20)
    mix = spk1 + spk2
    mix, clean, regi = unify_energy(mix, spk1, regi)
    return (mix, clean, regi)

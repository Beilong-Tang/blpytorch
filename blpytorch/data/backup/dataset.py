import torch
import torchaudio
import random
import numpy as np
from torch.utils.data import Dataset
import os.path as op

import random
from .backup import helper as h
import dac

from einops import rearrange

from .effect import _unify_energy


def clip_wav(mix_audio, clean_audio, length):
    """
    mix_audio and clean audio have shape [T]
    clip the audio to the certain length,
    """
    if length == None:
        return mix_audio, clean_audio
    if mix_audio.size(0) < length:
        pad_tensor = torch.zeros(length - mix_audio.size(0))
        clean_audio = torch.cat([clean_audio, pad_tensor])
        mix_audio = torch.cat([mix_audio, pad_tensor])
    elif mix_audio.size(0) > length:
        mix_audio = mix_audio[:length]
        clean_audio = clean_audio[:length]
    return mix_audio, clean_audio


def pad_emb(
    emb,
    length,
):
    """emb has shape [E, T']"""
    if length == None:
        return emb

    diff = length - emb.shape[-1]
    if diff < 0:
        return emb[:, :length]
    while diff > 0:
        pad = emb[:, :diff]
        emb = torch.concat([emb, pad], dim=-1)
        if emb.shape[-1] == length:
            return emb
        diff = length - emb.shape[-1]
    return emb


def clip_wav_tgt(mix_audio, tgt_audio, clean_audio, length):
    """
    mix_audio and clean audio have shape [T]
    clip the audio to the certain length,
    """
    if length == None:
        return mix_audio, tgt_audio, clean_audio
    if tgt_audio.size(0) > length:
        tgt_audio = tgt_audio[:length]
    elif tgt_audio.size(0) < length:
        pad_tensor = torch.zeros(length - tgt_audio.size(0))
        tgt_audio = torch.cat([tgt_audio, pad_tensor])

    if mix_audio.size(0) < length:
        pad_tensor = torch.zeros(length - mix_audio.size(0))
        clean_audio = torch.cat([clean_audio, pad_tensor])
        mix_audio = torch.cat([mix_audio, pad_tensor])
    elif mix_audio.size(0) > length:
        mix_audio = mix_audio[:length]
        clean_audio = clean_audio[:length]
    return mix_audio, tgt_audio, clean_audio


CURRENT = 0
NUM = 2


class LibriMixDataset(Dataset):
    def __init__(
        self,
        mix_path,
        source_path,
        length,
        mode="noise",
        input_type="wav",
        dac_audio_len=None,
        rank="",
        status="training",
    ):
        """
        Here the mix_length and source_path is the same, the source_path does not matter here due to the quality of the dataset.
        mode can be either noise or target
        input_type can be wav/dac/emb
        status can either be training or inference
        """
        self.mode = mode
        print(f"Libri mode is {mode}")
        print(f"input type is {input_type}")
        self.mix = []
        self.source = []
        self.input_type = input_type
        with open(mix_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                self.mix.append(l.replace("\n", "").split(" ")[-1])
        with open(source_path, "r") as f:
            lines = f.readlines()
            for l in lines:
                self.source.append(l.replace("\n", "").split(" ")[-1])
        self.length = length
        self.dac_audio_len = dac_audio_len
        self.rank = rank
        pass

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        input_type = self.input_type
        if input_type == "emb":
            if self.mode == "noise":
                mix_emb = np.load(self.mix[idx])  # [1, 1024, T']
                clean_emb = np.load(self.source[idx])  # [1, 1024, T']
                mix_emb = mix_emb[:, :, : self.length]
                clean_emb = clean_emb[:, :, : self.length]
                mix_emb = torch.from_numpy(mix_emb[0])  # [1024, T']
                clean_emb = torch.from_numpy(clean_emb[0])  # [1024, T']
                mix_emb, clean_emb = (
                    pad_emb(mix_emb, self.length),
                    pad_emb(clean_emb, self.length),
                )
                return mix_emb, clean_emb
            else:
                raise NotImplementedError("not implemented")
        elif input_type == "wav":
            if self.mode == "noise":
                mix_audio, _ = torchaudio.load(self.mix[idx])
                clean_audio, _ = torchaudio.load(self.source[idx])
                mix_audio = rearrange(mix_audio, "1 t -> t")
                clean_audio = rearrange(clean_audio, "1 t -> t")
                mix_audio, clean_audio = clip_wav(mix_audio, clean_audio, self.length)
                mix_audio, clean_audio = _unify_energy(mix_audio, clean_audio)
                global CURRENT, NUM
                if CURRENT < NUM:
                    torchaudio.save(
                        f"{self.rank}_noise_{idx}.wav", mix_audio.unsqueeze(0), 16000
                    )
                    torchaudio.save(
                        f"{self.rank}_clean_{idx}.wav", clean_audio.unsqueeze(0), 16000
                    )
                    CURRENT += 1
                return mix_audio, clean_audio
            elif self.mode == "target":
                mix_audio, _ = torchaudio.load(self.mix[idx])
                clean_audio, _ = torchaudio.load(self.source[idx])
                tgt_audio, _ = torchaudio.load(random.choice(self.source))
                mix_audio = rearrange(mix_audio, "1 t -> t")
                clean_audio = rearrange(clean_audio, "1 t -> t")
                tgt_audio = rearrange(tgt_audio, "1 t -> t")[: self.length]
                return clip_wav_tgt(mix_audio, tgt_audio, clean_audio, self.length)
        elif input_type == "dac":
            if self.mode == "noise":
                mix_dac, clean_dac = (
                    dac.DACFile.load(self.mix[idx]),
                    dac.DACFile.load(self.source[idx]),
                )
                mix_code, clean_code = (
                    mix_dac.codes[0],
                    clean_dac.codes[0],
                )  # [12 (codebook number), T]
                mix_db, clean_db = (
                    torch.from_numpy(mix_dac.input_db),
                    torch.from_numpy(clean_dac.input_db),
                )  # [1]

                mix_code, clean_code = (
                    pad_emb(mix_code, self.length),
                    pad_emb(clean_code, self.length),
                )

                return mix_code, mix_db, clean_code, clean_db, self.dac_audio_len
                pass
            else:
                raise NotImplementedError("not implemented")
        else:
            raise NotImplementedError(
                f"input_type can only be in [wav, dac, emb]. not {input_type}"
            )


if __name__ == "__main__":
    random.seed(1234)
    import yaml

    config_path = ""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pass

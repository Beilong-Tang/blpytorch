import os.path as op
import os
import sys
from torch.utils.data import Dataset
import random
import torchaudio
import torch
import torch.nn.functional as F
import random
import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.load_scp import get_source_list
from data.backup.helper import truc_wav


def generate_target_audio(spk1, spk2, regi, mix_length, regi_length, snr=5):
    """
    spk 1: T1
    spk 2: T2
    regi: T3
    """
    spk1, spk2 = _unify_energy(spk1, spk2)
    snr_1 = random.random() * snr / 2
    snr_2 = -snr_1
    spk1 = spk1 * 10 ** (snr_1 / 20)
    spk2 = spk2 * 10 ** (snr_2 / 20)
    mix = spk1 + spk2
    mix, clean, regi = _unify_energy(mix, spk1, regi)
    return (
        mix,
        clean,
        regi,
    )


class LibriMixTestDataset(Dataset):
    def __init__(
        self,
        mix_path: str,
        regi_path: str,
        clean_path: str,
        mix_length=48080,
        regi_length=64080,
    ):
        self.mix_list = get_source_list(mix_path)
        self.regi_list = get_source_list(regi_path)
        self.clean_list = get_source_list(clean_path)
        self.mix_length = mix_length
        self.regi_length = regi_length
        pass

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix_path = self.mix_list[idx]
        regi_path = self.regi_list[idx]
        clean_path = self.clean_list[idx]
        mix_audio = torchaudio.load(mix_path)[0].squeeze(0)  # [T]
        regi_audio = torchaudio.load(regi_path)[0].squeeze(0)
        clean_audio = torchaudio.load(clean_path)[0].squeeze(0)
        mix_audio, clean_audio = truc_wav(
            mix_audio, clean_audio, length=self.mix_length
        )
        regi_audio = truc_wav(regi_audio, length=self.regi_length)
        return mix_audio, clean_audio, regi_audio

    def test(self, output_dir, num=20):
        for i in tqdm.tqdm(range(num)):
            idx = random.randint(0, self.__len__())
            mix_audio, clean_audio, regi_audio = self.__getitem__(idx)
            torchaudio.save(
                op.join(output_dir, f"{i}_mix.wav"), mix_audio.unsqueeze(0), 16000
            )
            torchaudio.save(
                op.join(output_dir, f"{i}_clean.wav"), clean_audio.unsqueeze(0), 16000
            )
            torchaudio.save(
                op.join(output_dir, f"{i}_regi.wav"), regi_audio.unsqueeze(0), 16000
            )
        print(f"Testing is done for num {num}, audio output {output_dir}")


if __name__ == "__main__":
    from utils.io import make_path

    output_dir = make_path(
        "/public/home/qinxy/bltang/ml_framework_slurm/data_sample/output/libri2mix_test",
        is_dir=True,
    )
    libri = LibriMixTestDataset(
        "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/mix.scp",
        "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/bltang/regi.scp",
        "/public/home/qinxy/bltang/data/LibriMix/Libri2Mix/wav16k/min/lists/test/s1.scp",
    )
    libri.test(output_dir, num=20)

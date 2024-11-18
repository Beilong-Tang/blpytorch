from torch.utils.data import Dataset
import torch
import random
import tqdm
import torchaudio
import os.path as op
import sys
import os

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from .helper import truc_wav, generate_target_audio
from ..utils.load_scp import get_source_list


class AbsBlindDataset(Dataset):
    def test(self, output_dir, num=20):
        """
        Test method to test the dataset
        """
        for i in tqdm.tqdm(range(num)):
            idx = random.randint(0, self.__len__())
            mix_audio, tgt_1, tgt_2, _, _, _ = self.__getitem__(idx)
            torchaudio.save(
                op.join(output_dir, f"{i}_mix.wav"), mix_audio.unsqueeze(0), 16000
            )
            torchaudio.save(
                op.join(output_dir, f"{i}_1.wav"), tgt_1.unsqueeze(0), 16000
            )
            torchaudio.save(
                op.join(output_dir, f"{i}_2.wav"), tgt_2.unsqueeze(0), 16000
            )
        print(f"Testing is done for num {num}, audio output {output_dir}")


class BlindDataset(AbsBlindDataset):
    def __init__(
        self,
        mix_path: str,
        s1_path: str,
        s2_path: str,
        rank: int,
        mix_length=64000,
    ):
        """
        The regular dataset for target speaker extraction.
        Has to provide three .scp files that have mix_path, regi_path, clean_path aligned
        """
        self.mix_list = get_source_list(mix_path)
        self.s2_list = get_source_list(s2_path)
        self.s1_list = get_source_list(s1_path)
        self.mix_length = mix_length
        self.rank = rank

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        mix_path = self.mix_list[idx]
        s1_path = self.s1_list[idx]
        s2_path = self.s2_list[idx]
        mix_audio = torchaudio.load(mix_path)[0].squeeze(0)  # [T]
        s1_audio = torchaudio.load(s1_path)[0].squeeze(0)
        s2_audio = torchaudio.load(s2_path)[0].squeeze(0)
        mix_audio, s1_audio, s2_audio = truc_wav(
            mix_audio, s1_audio, s2_audio, length=self.mix_length
        )
        return mix_audio, s1_audio, s2_audio, mix_path, s1_path, s2_path

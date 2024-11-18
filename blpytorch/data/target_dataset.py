from torch.utils.data import Dataset
import torch
import random
import tqdm
import torchaudio
import os.path as op
import sys
import os
import numpy as np

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from data.helper import truc_wav, generate_target_audio
from utils.load_scp import get_source_list

SEED = 1234


class AbsTargetDataSet(Dataset):
    def test(self, output_dir, num=20):
        """
        Test method to test the dataset
        """
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


class TargetDMDataset(AbsTargetDataSet):
    def __init__(
        self,
        scp_path,
        rank,
        epoch_num=100000,
        mix_length=48080,
        regi_length=64080,
    ):
        """
        Initialize the Target DM Dataset.
        This class is used for dynamic mixing of target speech extraction dataset


        Args:
            scp_path: the .pt file which saves a dictionary of speker_name -> list of path to source files
            epoch_num: specifcy how many data to be considered as one epoch
            mix_length: the length of the mixing speech and clean speech
            regi_length: the length of the register speech
        """
        self.speaker_dict = torch.load(scp_path)
        self.length = epoch_num
        self.mix_length = mix_length
        self.rank = rank
        self.regi_length = regi_length
        self.num = 3
        self.ct = 0
        pass

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        keys_list = list(self.speaker_dict.keys())
        speaker_1 = random.choice(keys_list)
        speaker_2 = random.choice(keys_list)
        if self.ct < self.num:
            print(f"rank {self.rank} get spk1 {speaker_1}")
            self.ct += 1
        while speaker_2 == speaker_1:
            speaker_2 = random.choice(keys_list)
        spk1 = random.choice(self.speaker_dict[speaker_1])
        regi = random.choice(self.speaker_dict[speaker_1])
        while regi == spk1:
            regi = random.choice(self.speaker_dict[speaker_1])
        spk2 = random.choice(self.speaker_dict[speaker_2])
        spk1_audio = torchaudio.load(spk1)[0].squeeze(0)  # [T]
        spk2_audio = torchaudio.load(spk2)[0].squeeze(0)
        regi_audio = torchaudio.load(regi)[0].squeeze(0)
        if self.regi_length is not None:
            regi_audio = truc_wav(regi_audio, length=self.regi_length)
        else:
            regi_audio = truc_wav(regi_audio, length=self.mix_length)
        spk1_audio = truc_wav(spk1_audio, length=self.mix_length)
        spk2_audio = truc_wav(spk2_audio, length=self.mix_length)
        mix, clean, regi = generate_target_audio(
            spk1_audio, spk2_audio, regi_audio, self.mix_length, self.regi_length
        )
        return mix, clean, regi


class TargetDataset(AbsTargetDataSet):
    def __init__(
        self,
        mix_path: str,
        regi_path: str,
        clean_path: str,
        rank: int,
        mix_length=48080,
        regi_length=64080,
        _type="audio",
    ):
        """
        The regular dataset for target speaker extraction.
        Has to provide three .scp files that have mix_path, regi_path, clean_path aligned
        _type:
            audio: standing for reading audio files
            npy: read numpy file
        """
        self.mix_key, self.mix_list = get_source_list(mix_path, ret_name=True)
        self.regi_key, self.regi_list = get_source_list(regi_path, ret_name=True)
        self.clean_key, self.clean_list = get_source_list(clean_path, ret_name=True)
        self.mix_length = mix_length
        self.regi_length = regi_length
        self.rank = rank
        self._type = _type
        pass

    def __len__(self):
        return len(self.mix_list)

    def __getitem__(self, idx):
        name = self.mix_key[idx]
        regi_idx = self.regi_key.index(name)
        clean_idx = self.clean_key.index(name)
        mix_path = self.mix_list[idx]
        regi_path = self.regi_list[regi_idx]
        clean_path = self.clean_list[clean_idx]
        if self._type == "audio":
            mix = torchaudio.load(mix_path)[0].squeeze(0)  # [T]
            regi = torchaudio.load(regi_path)[0].squeeze(0)
            clean = torchaudio.load(clean_path)[0].squeeze(0)
        elif self._type == "npy":
            mix = torch.from_numpy(np.load(mix_path))  # [T, E]
            regi = torch.from_numpy(np.load(regi_path))
            clean = torch.from_numpy(np.load(clean_path))
            pass
        # print(f"before mix shape {mix.shape}, clean shape {clean.shape}, regi shape {regi.shape}, clean_path {clean_path}, mix_path {mix_path}, regi_path {regi_path}")
        mix, clean = truc_wav(mix, clean, length=self.mix_length)
        regi = truc_wav(regi, length=self.regi_length)
        # print(f"after mix shape {mix.shape}, clean shape {clean.shape}, regi shape {regi.shape}")
        return mix, clean, regi, mix_path, clean_path, regi_path

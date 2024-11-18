import random
import numpy as np
from scipy.io import wavfile
import scipy.signal as s
import warnings
import torch
import torch.nn.functional as F


def resample(audio, original_fs, target_fs):
    return s.resample_poly(audio, target_fs, original_fs)


def uniform_sample(
    range=None, lower=None, upper=None, prob: bool = None, return_int=False
):
    if range is not None and len(range) == 2:
        return _random_uniform(range, return_int)
    elif lower == None or upper == None:
        k = random.uniform(0, 1)
        return k < prob
    else:
        return _random_uniform([lower, upper], return_int)


def _random_uniform(range, return_int=False):
    assert len(range) == 2
    if return_int:
        return random.randint(*range)
    else:
        return random.uniform(*range)


def readfiles(file_path):
    res = []
    language = []
    return_lan = False
    with open(file_path, "r") as f:
        for l in f.readlines():
            contents = l.replace("\n", "").split(" ")
            if len(contents) == 3:
                if not return_lan:
                    return_lan = True
                language.append(contents[-1])
                res.append(contents[1])
            else:
                res.append(contents[-1])
    if return_lan:
        return res, language
    else:
        return res


def random_choice(list):
    return random.choice(list)


warnings.filterwarnings("ignore")


def readwav(file_path, fs, normalize=True) -> np.number:
    wav_fs, wav = wavfile.read(file_path)
    if len(wav.shape) == 2:
        print(f"{file_path} is a multi-channel audio")
        wav = wav[:, 0]
    if normalize:
        if np.max(np.abs(wav)) == 0:
            print(f"audio {file_path} is no voice")
        else:
            wav = wav / np.max(np.abs(wav))
    try:
        assert len(wav.shape) == 1
    except:
        print(file_path)
    if wav_fs != fs:
        wav = resample(wav, wav_fs, fs)
    if normalize:
        if np.max(np.abs(wav)) == 0:
            print(f"audio {file_path} is no voice")
        else:
            wav = wav / np.max(np.abs(wav))
    return wav


def random_trunk(frame, frame_length, sr, start=None, return_duration=False):
    """
    frame_length : audio duration in seconds

    """
    # [samples, channel]
    trunk, length = None, 0
    while length - frame_length < -0.1:
        trunk_length = frame_length - length
        segment, duration, sr, start_sec, end_sec = _random_chunk_wav_file(
            frame, trunk_length, sr, start
        )
        length += duration
        if trunk is None:
            trunk = segment
        else:
            trunk = np.concatenate([trunk, segment], axis=0)
    if return_duration:
        return trunk, start_sec, end_sec
    else:
        return trunk


def _random_chunk_wav_file(frame, chunk_length, sr, start=None):
    """
    fname: 1d audio file
    chunk_length : the length to chunk in seconds
    sr: the sample rate of the frame
    if the chunk_length >  duration of the frame, then just return the frame and its length
    else if the chunk_length < duration of the frame, then we chunk the audio
    start: the starting second of the audio
    """
    sample_length = len(frame)
    duration = sample_length / sr
    if duration < chunk_length or abs(duration - chunk_length) < 1e-4:
        ## if the audio is shorter than the chunk_length, just return the audio
        return frame.copy(), duration, sr, 0, duration  # [-1,1]
    else:
        # Random trunk
        if start == None:
            random_starts = np.random.randint(0, sample_length - sr * chunk_length)
        else:
            random_starts = int(start * sr)
        random_end = int(random_starts + sr * chunk_length)  ## points
        if random_end > sample_length:
            random_end = sample_length
        frames = frame[random_starts:random_end]
        frames = constrain_length(frames, length=int(chunk_length * sr))
        return frames, chunk_length, sr, random_starts / sr, random_end / sr


def constrain_length(chunk, length):
    """
    chunk: 1d numpy audio
    length: point length
    """
    frames_length = chunk.shape[0]
    if frames_length == length:
        return chunk
    elif frames_length < length:
        return np.pad(chunk, ((0, int(length - frames_length))), "constant")
    else:
        return chunk[:length]


if __name__ == "__main__":
    x = np.random.rand(30)
    print(x)
    a = constrain_length(x, 40)
    print(a)
    print(a.shape)

    rate, a = wavfile.read("/home/bltang/work/data/test/p227_001_mic1.wav")

    print(a.shape)
    frame = random_trunk(a, 6, rate)
    wavfile.write("out.wav", rate, frame)
    print(frame.shape)

    pass

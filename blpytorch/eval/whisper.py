"""
Evaluate wer using whisper
"""

import whisper
import jiwer
import torch

class WhipserWrapper:

    def __init__(self, _type = "base", language = "english", device = "cuda:0"):
        self.model = whisper.load_model(_type, device = device)
        pass

    def wer_batch(self, output, true, sample_rate = 0.1):
        """
        Args:
            Calculate the wer within batches
            output: [B, T]
            true: [B, T]
            sample_rate: controls how many audios to calculate wer
        Returns:
            wer: float
        """
        n = int(len(output) * sample_rate)
        indices = torch.randperm(len(output))[:n]
        output = output[indices] # [B',T]
        true = true[indices] #[B',T]
        output_transcript = self.transcribe_batch(output)
        true_transcript = self.transcribe_batch(true)
        wer = jiwer.wer(true_transcript,output_transcript)
        return wer

    def transcribe_batch(self, audio):
        """
        Output transcript of audio 
        audio: [B,T]
        """
        res = []
        for _audio in audio: # [T]
            _audio = whisper.pad_or_trim(_audio)
            mel = whisper.log_mel_spectrogram(_audio).to(self.model.device)
            # decode the audio
            result = whisper.decode(self.model, mel)
            text = result.text
            res.append(text)
        return res

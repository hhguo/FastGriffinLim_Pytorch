import fire
import numpy as np
import time
import torch

from audio import TorchSTFT, load_wav, save_wav


def main(file_in, file_out):
    Convertor =  TorchSTFT(2048, 300, 1200, 80, 24000)

    if '.wav' in file_in:
        wav_in = load_wav(file_in, 24000)
        wav_in = torch.Tensor(wav_in).view(1, -1)
        spec = Convertor.mel_spectrogram(wav_in)
    elif '.npy' in file_in:
        spec = torch.from_numpy(np.load(file_in)).unsqueeze(0).transpose(1, 2)

    wav_out = Convertor.inv_mel_spectrogram(spec)
    #print(t2 - t1, wav_in.shape[-1] / 24000.)

    wav_out = wav_out[0].numpy()
    save_wav(wav_out, file_out, 24000)


if __name__ == '__main__':
    fire.Fire(main)
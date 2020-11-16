from librosa.util import pad_center, tiny
from scipy.io.wavfile import write
from scipy.signal import get_window
from torch import Tensor
from torch.autograd import Variable
from typing import Optional, Tuple

import librosa
import librosa.util as librosa_util
import math
import numpy as np
import scipy
import torch
import torch.nn.functional as F
import warnings


def load_wav(path, sample_rate):
    waveform = librosa.core.load(path, sr=sample_rate)[0]
    return waveform


def save_wav(wav, path, sample_rate):
    wav *= 32767.0
    write(path, sample_rate, wav.astype(np.int16))


class TorchSTFT(torch.nn.Module):
    def __init__(self, fft_size, hop_size, win_size, mel_size,
                 sample_rate=24000,
                 preemphasis=0.97,
                 normalized=False,
                 ref_level_db=20,
                 min_level_db=-115):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.mel_size = mel_size
        self.ref_level_db = ref_level_db
        self.min_level_db = min_level_db
        self.normalized = normalized
        self.sample_rate = sample_rate
        self.window = torch.hann_window(win_size)
        self.mel_scale = MelScale(n_mels=mel_size,
                                  sample_rate=sample_rate,
                                  n_stft=(fft_size // 2 + 1))

    def mel_spectrogram(self, x):
        # STFT transform
        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_size,
                            self.window.type_as(x), normalized=self.normalized)
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        # complex to magphase
        mag = torch.clamp(real ** 2 + imag ** 2, min=1e-7)
        mag = torch.sqrt(mag)
        phase = torch.atan2(imag, real)
        
        # Mel Scale
        mag = self.mel_scale(mag)
        
        # dB Scale
        mag = 20 * torch.log10(mag) - self.ref_level_db
        mag = torch.clamp((mag - self.min_level_db) / -self.min_level_db, 0, 1)

        return mag

    def inv_mel_spectrogram(self, mag):
        # Inverse Log Scale
        mag = torch.clamp(mag, 0, 1) * -self.min_level_db + self.min_level_db
        mag = torch.pow(10, (mag + self.ref_level_db) / 20)
        
        # Inverse Mel Scale
        mag = self.mel_scale.inverse(mag)

        # griffin lim
        waveform = self.fast_griffinlim(mag)
        #waveform = self.griffinlim(mag)

        return waveform

    def fast_griffinlim(self, specgram,
                        power=1.0, n_iter=10, momentum=0.99, length=None, rand_init=True):
        assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
        assert momentum >= 0, 'momentum={} < 0'.format(momentum)
        
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape([-1] + list(shape[-2:]))

        specgram = specgram.pow(power)

        # randomly initialize the phase
        batch, freq, frames = specgram.size()
        if rand_init:
            angles = 2 * math.pi * torch.rand(batch, freq, frames)
        else:
            angles = torch.zeros(batch, freq, frames)
        angles = torch.stack([angles.cos(), angles.sin()], dim=-1) \
            .to(dtype=specgram.dtype, device=specgram.device)

        specgram = specgram.unsqueeze(-1).expand_as(angles)
        t_0 = specgram * angles
        inverse = torch.istft(t_0,
                        n_fft=self.fft_size,
                        hop_length=self.hop_size,
                        win_length=self.win_size,
                        window=self.window,
                        normalized=self.normalized,
                        length=length).float()

        for i in range(n_iter):
            # Rebuild the spectrogram
            rebuilt = torch.stft(inverse, self.fft_size, self.hop_size, self.win_size, self.window,
                                True, 'reflect', self.normalized, True)
            angles = torch.atan2(rebuilt[..., 1], rebuilt[..., 0])
            angles = torch.stack([angles.cos(), angles.sin()], dim=-1).to(dtype=specgram.dtype, device=specgram.device)

            # Update our phase estimates
            t_1 = specgram * angles
            c = t_1 + momentum * (t_1 - t_0)
            t_0 = t_1
            # Invert with our current estimate of the phases
            inverse = torch.istft(c,
                                  n_fft=self.fft_size,
                                  hop_length=self.hop_size,
                                  win_length=self.win_size,
                                  window=self.window,
                                  normalized=self.normalized,
                                  length=length).float()

        # unpack batch
        inverse = inverse.reshape(shape[:-2] + inverse.shape[-1:])

        return inverse

    def griffinlim(self, specgram,
                   power=1.0, n_iter=15, momentum=0.99, length=None, rand_init=True):
        assert momentum < 1, 'momentum={} > 1 can be unstable'.format(momentum)
        assert momentum >= 0, 'momentum={} < 0'.format(momentum)

        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape([-1] + list(shape[-2:]))

        specgram = specgram.pow(power)

        # randomly initialize the phase
        batch, freq, frames = specgram.size()
        if rand_init:
            angles = 2 * math.pi * torch.rand(batch, freq, frames)
        else:
            angles = torch.zeros(batch, freq, frames)
        angles = torch.stack([angles.cos(), angles.sin()], dim=-1) \
            .to(dtype=specgram.dtype, device=specgram.device)
        specgram = specgram.unsqueeze(-1).expand_as(angles)

        # And initialize the previous iterate to 0
        rebuilt = torch.tensor(0.)

        for _ in range(n_iter):
            # Store the previous iterate
            tprev = rebuilt

            # Invert with our current estimate of the phases
            inverse = torch.istft(specgram * angles,
                                  n_fft=self.fft_size,
                                  hop_length=self.hop_size,
                                  win_length=self.win_size,
                                  window=self.window,
                                  normalized=self.normalized,
                                  length=length).float()

            # Rebuild the spectrogram
            rebuilt = torch.stft(inverse, self.fft_size, self.hop_size, self.win_size, self.window,
                                True, 'reflect', self.normalized, True)

            # Update our phase estimates
            angles = rebuilt
            if momentum:
                angles = angles - tprev.mul_(momentum / (1 + momentum))
            angles = angles.div(complex_norm(angles).add(1e-16).unsqueeze(-1).expand_as(angles))

        # Return the final phase estimates
        waveform = torch.istft(specgram * angles,
                               n_fft=self.fft_size,
                               hop_length=self.hop_size,
                               win_length=self.win_size,
                               window=self.window,
                               normalized=self.normalized,
                               length=length).float()

        # unpack batch
        waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

        return waveform

    def preemphasis(self, x):
        return lfilter(x, [1, -self.alpha], [1])


    def inv_preemphasis(self, x):
        return lfilter(x, [1], [1, -self.alpha])



class MelScale(torch.nn.Module):
    r"""Turn a normal STFT into a mel frequency STFT, using a conversion
    matrix.  This uses triangular filter banks.

    User can control which device the filter bank (`fb`) is (e.g. fb.to(spec_f.device)).

    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
            if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
    """
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 n_mels: int = 128,
                 sample_rate: int = 24000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None) -> None:
        super(MelScale, self).__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        assert f_min <= self.f_max, 'Require f_min: %f < f_max: %f' % (f_min, self.f_max)

        #fb = torch.empty(0) if n_stft is None else create_fb_matrix(
        #    n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate)
        fb = torch.from_numpy(librosa.filters.mel(sample_rate,
                                                  (n_stft - 1) * 2,
                                                  htk=False,
                                                  n_mels=n_mels)).T
        ifb = torch.pinverse(fb, rcond=1e-8)

        self.register_buffer('fb', fb)
        self.register_buffer('ifb', ifb)

    def forward(self, specgram: Tensor) -> Tensor:
        r"""
        Args:
            specgram (Tensor): A spectrogram STFT of dimension (..., freq, time).

        Returns:
            Tensor: Mel frequency spectrogram of size (..., ``n_mels``, time).
        """

        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(specgram.size(1), self.f_min, self.f_max, self.n_mels, self.sample_rate)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])

        return mel_specgram

    def inverse(self, melspec):
        mag = torch.matmul(melspec.transpose(1, 2), self.ifb).transpose(1, 2).clamp(min=1e-10)
        return mag


def complex_norm(
        complex_tensor: Tensor,
        power: float = 1.0
) -> Tensor:
    r"""Compute the norm of complex tensor input.

    Args:
        complex_tensor (Tensor): Tensor shape of `(..., complex=2)`
        power (float): Power of the norm. (Default: `1.0`).

    Returns:
        Tensor: Power of the normed input tensor. Shape of `(..., )`
    """

    # Replace by torch.norm once issue is fixed
    # https://github.com/pytorch/pytorch/issues/34279
    return complex_tensor.pow(2.).sum(-1).pow(0.5 * power)


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        norm: Optional[str] = None
) -> Tensor:
    r"""Create a frequency bin conversion matrix.

    Args:
        n_freqs (int): Number of frequencies to highlight/apply
        f_min (float): Minimum frequency (Hz)
        f_max (float): Maximum frequency (Hz)
        n_mels (int): Number of mel filterbanks
        sample_rate (int): Sample rate of the audio waveform
        norm (Optional[str]): If 'slaney', divide the triangular mel weights by the width of the mel band
        (area normalization). (Default: ``None``)

    Returns:
        Tensor: Triangular filter banks (fb matrix) of size (``n_freqs``, ``n_mels``)
        meaning number of frequencies to highlight/apply to x the number of filterbanks.
        Each column is a filterbank so that assuming there is a matrix A of
        size (..., ``n_freqs``), the applied result would be
        ``A * create_fb_matrix(A.size(-1), ...)``.
    """

    if norm is not None and norm != "slaney":
        raise ValueError("norm must be one of None or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
    m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
    m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
    f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
    # create overlapping triangles
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.min(down_slopes, up_slopes)
    fb = torch.clamp(fb, 1e-6, 1)

    if norm is not None and norm == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)
    return fb


def lfilter(waveform, a_coeffs, b_coeffs, clamp=True):
    # pack batch
    shape = waveform.size()
    waveform = waveform.reshape(-1, shape[-1])

    a_coeffs = torch.Tensor(a_coeffs)
    b_coeffs = torch.Tensor(b_coeffs)

    assert (a_coeffs.size(0) == b_coeffs.size(0))
    assert (len(waveform.size()) == 2)
    assert (waveform.device == a_coeffs.device)
    assert (b_coeffs.device == a_coeffs.device)

    device = waveform.device
    dtype = waveform.dtype
    n_channel, n_sample = waveform.size()
    n_order = a_coeffs.size(0)
    n_sample_padded = n_sample + n_order - 1
    assert (n_order > 0)

    # Pad the input and create output
    padded_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)
    padded_waveform[:, (n_order - 1):] = waveform
    padded_output_waveform = torch.zeros(n_channel, n_sample_padded, dtype=dtype, device=device)

    # Set up the coefficients matrix
    # Flip coefficients' order
    a_coeffs_flipped = a_coeffs.flip(0)
    b_coeffs_flipped = b_coeffs.flip(0)

    # calculate windowed_input_signal in parallel
    # create indices of original with shape (n_channel, n_order, n_sample)
    window_idxs = torch.arange(n_sample, device=device).unsqueeze(0) + torch.arange(n_order, device=device).unsqueeze(1)
    window_idxs = window_idxs.repeat(n_channel, 1, 1)
    window_idxs += (torch.arange(n_channel, device=device).unsqueeze(-1).unsqueeze(-1) * n_sample_padded)
    window_idxs = window_idxs.long()
    # (n_order, ) matmul (n_channel, n_order, n_sample) -> (n_channel, n_sample)
    input_signal_windows = torch.matmul(b_coeffs_flipped, torch.take(padded_waveform, window_idxs))

    input_signal_windows.div_(a_coeffs[0])
    a_coeffs_flipped.div_(a_coeffs[0])
    for i_sample, o0 in enumerate(input_signal_windows.t()):
        windowed_output_signal = padded_output_waveform[:, i_sample:(i_sample + n_order)]
        o0.addmv_(windowed_output_signal, a_coeffs_flipped, alpha=-1)
        padded_output_waveform[:, i_sample + n_order - 1] = o0

    output = padded_output_waveform[:, (n_order - 1):]

    if clamp:
        output = torch.clamp(output, min=-1., max=1.)

    # unpack batch
    output = output.reshape(shape[:-1] + output.shape[-1:])

    return output
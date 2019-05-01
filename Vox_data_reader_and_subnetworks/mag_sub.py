
#from vox2_data_reader import ModelConfig
import librosa
import numpy as np
from model import ModelConfig

def get_stft_matrix(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)


def mag_process(mag, len_frame, len_hop, num_iters, phase_angle=None, length=None):
    assert(num_iters > 0)
    if phase_angle is None:
        phase_angle = np.pi * np.random.rand(*mag.shape)
    spec = get_stft_matrix(mag, phase_angle)
    for i in range(num_iters):
        wav = librosa.istft(spec, win_length=len_frame, hop_length=len_hop, length=length)
        if i != num_iters - 1:
            spec = librosa.stft(wav, n_fft=len_frame, win_length=len_frame, hop_length=len_hop)
            _, phase = librosa.magphase(spec)
            phase_angle = np.angle(phase)
            spec = get_stft_matrix(mag, phase_angle)
    return wav


def to_wav_mag_only(mag, init_phase, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP, num_iters=50):
    #return np.array(list(map(lambda m_p: mag_process(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p)[0], list(zip(mag, init_phase))[1])))
    return np.array(list(map(lambda m: lambda p: mag_process(m, len_frame, len_hop, num_iters=num_iters, phase_angle=p), list(zip(mag, init_phase))[1])))


def magnitude_subnet(mag,video):
    print("START MAGNITUDE SUBNETWORK")
    return 1

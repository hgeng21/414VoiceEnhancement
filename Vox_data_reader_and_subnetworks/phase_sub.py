
from model import ModelConfig

def to_spectrogram(wav, len_frame=ModelConfig.L_FRAME, len_hop=ModelConfig.L_HOP):
    return np.array(list(map(lambda w: librosa.stft(w, n_fft=len_frame, hop_length=len_hop), wav)))


def phase_subnet(phase, mag_result):
    print("START PHASE SUBNETWORK")
    
    return mag_result
"""
load_audio.py
With functions to get
    1. front waveform
    2. back waveform
    3. 4th version of audio delay
And Audio class with methods to get pretrained vggish embedding
"""

import pandas as pd
import numpy as np
from csv import reader
from typing import Optional
from scipy import signal
from scipy.signal import hilbert, sosfiltfilt, butter
import librosa
import torch
import config

from src.vggish import vggish_input, vggish_pytorch
# from models.pytorch_vggish import pretrained
# PYTORCH_MODEL = pretrained.make_pytorch_vggish()
# PRETRAINED_VGGISH_TORCH = torch.load(config.PRETRAINED_VGGISH_PYTORCH_PATH)


def get_front_waveform(
    user_id: int,
    sr: Optional[str] = config.AUDIO_SAMPLING_RATE
) -> np.ndarray:
    """
    Get the front waveform data.

    :param user_id: an integer of user id
    :param sr: an integer of sampling rate
    :return: 1D np.array of the front waveform data
    """
    wav_front_fn = f'{config.DATAFRAME_PATH}/data_capture_{user_id}/front_audio_data.wav'
    x, sr = librosa.load(wav_front_fn, sr=sr)
    return x


def get_back_waveform(
    user_id: int,
    sr: Optional[str] = config.AUDIO_SAMPLING_RATE
) -> np.ndarray:
    """
    Get the back waveform data.

    :param user_id: an integer of user id
    :param sr: an integer of sampling rate
    :return: 1D np.array of the back waveform data
    """
    wav_back_fn = f'{config.DATAFRAME_PATH}/data_capture_{user_id}/back_audio_data.wav'
    x, sr = librosa.load(wav_back_fn, sr=sr)
    return x


def get_audio_delay4_clean(user_id: int) -> pd.Series:
    """
    Get the 4th version of audio delay. (written hastily)

    :param user_id: an integer of user id
    :return: a 1D pd.Series mapping from timestamp to audio delay
    """
    t_4 = []
    delay_4 = []
    filename = f'{config.DATAFRAME_PATH}/data_capture_{user_id}/bss_locate_spec.csv'
    with open(filename, 'r') as file_bss_locate_spec:
        read = reader(file_bss_locate_spec)
        for row in read:
            t_4.append(float(row[0]))
            delay_4.append(float(row[1]))

    delay_4 = np.array(delay_4)
    corr_btwn_mics = np.copy(delay_4)
    corr_btwn_mics[corr_btwn_mics > (-.35)] = 1
    corr_btwn_mics[corr_btwn_mics <= (-.35)] = 0
    delay_4[delay_4 < (-.35)] = 0
    delay_med_filt_sz = pd.Series(signal.medfilt(delay_4, 15))
    sos = butter(3, .07, output='sos')
    delay_med_filt_lowpass = sosfiltfilt(sos, delay_med_filt_sz)
    delay_med_filt_lowpass = pd.Series(delay_med_filt_lowpass)
    delay_med_filt_lowpass.index = t_4
    return delay_med_filt_lowpass


class Audio:
    """
    A class that integrates the front waveform, back waveform, and the
    vggish pre-trained audio embedding for a user id.
    """

    def __init__(
        self,
        user_id: int,
        sr: Optional[int] = config.AUDIO_SAMPLING_RATE,
        pretrained_path: Optional[str] = config.PRETRAINED_PATH
    ) -> None:
        self.user_id = user_id
        self.sr = sr
        self.front_frames = get_front_waveform(user_id, sr)
        self.back_frames = get_back_waveform(user_id, sr)
        self.pretrained_path = pretrained_path

    def get_pretrained_vggish(self):
        """
        Get pretrained vggish model in PyTorch. Load the
        parameters and import them into pytorch model
        structure.
        """
        pretrained_model = vggish_pytorch.VGGish()
        pretrained_model.load_state_dict(torch.load(self.pretrained_path))
        pretrained_model.eval()
        return pretrained_model

    def get_vggish_embedding(self) -> pd.DataFrame:
        """
        Get the vggish audio embedding for user user_id.
        :return: 2D pd.DataFrame of audio embedding, with
        shape = (NUM_FRAMES, )
        """
        input_batch = vggish_input.waveform_to_examples(
            self.front_frames, self.sr)
        input_batch = torch.from_numpy(input_batch).unsqueeze(dim=1)
        input_batch = input_batch.float()

        pretrained_model = self.get_pretrained_vggish()
        embedding = pretrained_model(input_batch).detach().numpy()
        embedding = pd.DataFrame(
            embedding,
            index=0.96 * np.arange(embedding.shape[0])
        )
        return embedding

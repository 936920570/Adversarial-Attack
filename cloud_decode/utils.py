# coding=utf-8
import warnings
import logging


def mute_third_party_logging():
    import os
    warnings.filterwarnings('ignore')
    spleeter_logger = logging.getLogger('spleeter')
    spleeter_logger.setLevel(logging.ERROR)
    sox_logger = logging.getLogger('sox')
    sox_logger.setLevel(logging.ERROR)
    tensorflow_logger = logging.getLogger("tensorflow")
    tensorflow_logger.setLevel(logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


mute_third_party_logging()

import os
import functools
import json
import re
import traceback
import wave
import hashlib
# import python_speech_features
import librosa
import numpy as np
import psutil
import multiprocessing
from collections import Iterable
from concurrent import futures
from enum import Enum
from typing import Tuple, List, Union, Any, Callable
from nltk.stem import WordNetLemmatizer
from progressbar import *

multiprocessing.set_start_method("spawn", force=True)

__file_folder__ = os.path.dirname(__file__)
__format_string__ = '%(asctime)s %(filename)s:%(lineno)d %(levelname)s: %(message)s'
__datefmt_string__ = '%Y-%m-%d %H:%M:%S'
__formatter__ = logging.Formatter(__format_string__, datefmt=__datefmt_string__)
logger = logging.getLogger("phantom-of-formant")
logger.propagate = False
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    __stream_handler__ = logging.StreamHandler()
    __stream_handler__.setFormatter(__formatter__)
    logger.addHandler(__stream_handler__)

EPS = 10 ** (-12)
MAX_FORMANT_NUM = 5
GOOGLE_JSON_SUFFIX = ".google.json"
AZURE_JSON_SUFFIX = ".azure.json"
OPPO_JSON_SUFFIX = ".oppo.json"
IBM_JSON_SUFFIX = ".ibm.json"
AMAZON_JSON_SUFFIX = ".amazon.json"
DEEPSPEECH_JSON_SUFFIX = ".deepspeech.json"
ANALYSIS_SUFFIX = ".pkl"
PICK_NPZ_SUFFIX = ".pick.npz"
IFLYTEC_JSON_SUFFIX = ".iflytec.json"
ALIYUN_JSON_SUFFIX = ".aliyun.json"
TENCENT_JSON_SUFFIX = ".tencent.json"
ACCOMPANIMENT_WAV_SUFFIX = ".accompaniment.wav"
FILT_WAV_SUFFIX = ".filt.wav"
VOCAL_WAV_SUFFIX = ".vocals.wav"
REVERB_WAV_SUFFIX = ".reverb.wav"
ECHO_WAV_SUFFIX = ".echo.wav"
TMP_WAV_SUFFIX = ".tmp.wav"
IRRELEVANT_WAV_SUFFIX = [ACCOMPANIMENT_WAV_SUFFIX, VOCAL_WAV_SUFFIX, REVERB_WAV_SUFFIX, ECHO_WAV_SUFFIX, TMP_WAV_SUFFIX, FILT_WAV_SUFFIX]
SCRIPT_VERSION = "1.1.0"

__stopwords__ = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                 "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                 "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                 "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                 "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                 "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                 "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "over",
                 "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
                 "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
                 "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
                 "should", "now"]
# Removed cn_stopwords.txt loading as the file doesn't exist


class PhoneType(Enum):
    Sil = 0  # Silence
    Voiceless = 1  # Voiceless
    Voice = 2  # Voiced
    Vowel = 3  # Vowel


class VoicelessProcessType(Enum):
    FormantFilter = 1  # Use formant filter
    BlueNoise = 2  # Blue noise
    WhiteNoise = 3  # White noise
    VioletNoise = 4  # Violet noise
    PinkNoise = 5  # Pink noise
    AdvancedNoise = 6  # Different processing for different voiceless sounds
    RedNoise = 7  # Red noise


class NoiseType(Enum):
    BlueNoise = 1  # Blue noise
    WhiteNoise = 2  # White noise
    VioletNoise = 3  # Violet noise
    PinkNoise = 4  # Pink noise
    RedNoise = 5  # Red noise


class OverflowProcessType(Enum):
    Clip = 1  # Clip
    Scale = 2  # Global scaling


class APIType(Enum):
    Azure = 1  # Microsoft Azure
    Aspire = 2  # Kaldi Aspire
    Google = 3  # Google Speech-to-Text
    IBM = 4  # IBM Speech to Text
    Amazon = 5  # Amazon Transcription
    Iflytec = 6  # xunfei
    Aliyun = 7
    Tencentyun = 8
    OPPO = 9


# task folder
def get_task_folder(task_name: str):
    return os.path.join("./task", task_name)


# for pick process
def get_command_folder(task_name):
    return os.path.join(get_task_folder(task_name), "command")


def get_perturbation_folder(task_name):
    return os.path.join(get_task_folder(task_name), "perturbation")


def get_wake_up_folder(task_name):
    return os.path.join(get_task_folder(task_name), "wake-up")


def get_music_folder(task_name):
    return os.path.join(get_task_folder(task_name), "music")


def get_pick_folder(task_name):
    return os.path.join(get_task_folder(task_name), 'pick')


def get_reselect_folder(task_name):
    return os.path.join(get_task_folder(task_name), "reselect")


def get_random_pick_folder(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), 'random-pick')


def get_pick_param_pkl_path(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), 'pick-params.pkl')


def get_pick_param_json_path(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), "pick-params.json")


# for generation process
def get_generate_folder(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), "generate")


# for intermediate success clips
def get_intermediate_folder(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), "intermediate")


# for visqol process
def get_visqol_folder(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), "visqol")


def get_separate_folder(task_name):
    return os.path.join(get_task_folder(task_name), 'separate')


# for api-find process
def get_find_folder(task_name: str) -> str:
    return os.path.join(get_task_folder(task_name), "find")


def get_adversarial_filename(clip_name: str, delta_db: Union[float, list], bandwidth: Union[int, list], precision: int = 1) -> str:
    if isinstance(delta_db, Iterable):
        logger.warning("You are using a Iterable value for delta_db. Since the adversarial name will be too long, it's better to use formant_weight to control a different reinforcement ratio for different formant.")
        delta_db_name = "[" + "_".join([("{:." + str(precision) + "f}").format(_db_) for _db_ in delta_db]) + "]"
    else:
        delta_db_name = ("{:." + str(precision) + "f}").format(delta_db)
    if isinstance(bandwidth, Iterable):
        bandwidth_name = "[" + "_".join(["{:d}".format(_b_) for _b_ in bandwidth]) + "]"
    else:
        bandwidth_name = "{:d}".format(bandwidth)
    return "{}_bw{}_db{}".format(clip_name, bandwidth_name, delta_db_name)


def path_join(*args):
    joined_path = ""
    for path_arg in args:
        if path_arg:
            joined_path = os.path.join(joined_path, path_arg)
    return joined_path


def wav_read(wav_path: str, mono: bool = True, expected_sr: int = None) -> Tuple[int, np.ndarray]:
    """
    read the wav file
    :return: (sample_rate, wav_signal)
    """
    wav_signal, sample_rate = librosa.load(wav_path, sr=expected_sr, mono=mono)
    wav_signal = wav_signal * 32768
    return sample_rate, wav_signal


def wav_write(audio_signal, file_path, sample_rate):
    """
    save sample
    :param file_path: the file path to save the audio signal
    :param audio_signal: audio signal
    :type audio_signal: np.ndarray
    :param sample_rate: sample rate, 8K or 16K
    :return: sample_path
    """
    _wv = wave.open(file_path, 'wb')
    _wv.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
    wv_data = audio_signal.astype(np.int16)
    wv_data = np.clip(wv_data, -32768, 32767)
    _wv.writeframes(wv_data.tobytes())
    _wv.close()


def normalize_signal(wav_signal: np.ndarray, scale: bool = True):
    wav_signal = np.double(wav_signal)
    wav_signal = wav_signal / (2.0 ** 15)
    dc_offset = np.mean(wav_signal)
    wav_signal = wav_signal - dc_offset
    if scale:
        signal_max = np.max(np.abs(wav_signal))
        wav_signal = wav_signal / (signal_max + EPS)  # normalize the signal to -1~1.

    return wav_signal


def frame_signal(wav_signal: np.ndarray, frame_len: int, step_len: int,
                 window_func: Callable = np.kaiser) -> np.ndarray:
    n_samples = len(wav_signal)
    n_frames = math.floor((n_samples - frame_len) * 1.0 / step_len) + 1
    if n_frames < 1:
        raise ValueError("The window size exceeds the length of wav_signal.")

    window = window_func(frame_len)
    x_index = np.arange(frame_len).reshape((1, -1))
    x_index = np.tile(x_index, (n_frames, 1))
    y_index = np.arange(n_frames).reshape((-1, 1)) * step_len
    y_index = np.tile(y_index, (1, frame_len))
    framing_index = x_index + y_index
    framed_signal = wav_signal[framing_index]

    return framed_signal * window


def cal_spectrum(wav_signal: np.ndarray, frame_len: int, step_len: int) -> np.ndarray:
    """
    calculate the spectrum of signal.
    :param wav_signal: wav signal
    :param frame_len: the length of frame
    :param step_len: the length of step
    :return: spectrum with shape (frame_num, frame_len/2+1)
    """
    framed_signal = frame_signal(wav_signal, frame_len, step_len, np.ones)
    return np.fft.rfft(framed_signal)


# def cal_power_spectrum(wav_signal: np.ndarray, frame_len: int, step_len: int, pre_emphasis: bool = True, window_func: Callable = np.hamming):
#     """
#     calculate the power spectrum of signal.
#     :param step_len: the length of step.
#     :param frame_len: the length of frame.
#     :param wav_signal: wav signal.
#     :param pre_emphasis: whether add pre-emphasis on signal. Default is True.
#     :param window_func: the window function like hinning window. Default is np.hamming
#     return: power_spec with shape (frame_num, frame_len/2+1)
#     """
#     if pre_emphasis:
#         wav_signal = python_speech_features.sigproc.preemphasis(wav_signal)
#     framed_signal = frame_signal(wav_signal, frame_len, step_len, window_func=window_func)
#     power_spec = python_speech_features.sigproc.powspec(framed_signal, frame_len)
#     return power_spec  # shape: (frame_num, fft_num)


# def file_2_power_spectrum(wav_file: str, frame_time: float, step_time: float, pre_emphasis: bool = True, window_func: Callable = np.hamming):
#     """
#     calculate the power spectrum of signal. (The signal is -32768~32767)
#     :param wav_file: wav signal.
#     :param step_time: the length of step.
#     :param frame_time: the length of frame.
#     :param pre_emphasis: whether add pre-emphasis on signal. Default is True.
#     :param window_func: the window function like hinning window. Default is np.hamming
#     return: power_spec with shape (frame_num, frame_len/2+1)
#     """
#     sample_rate, wav_signal = wav_read(wav_file)
#     frame_len = int(frame_time * sample_rate)
#     step_len = int(step_time * sample_rate)
#     return cal_power_spectrum(wav_signal, frame_len, step_len, pre_emphasis, window_func)


# def get_frame_energy_list(wav_file: str, frame_time: float, step_time: float, pre_emphasis: bool = True, window_func: Callable = np.hamming) -> np.ndarray:
#     sample_rate, wav_signal = wav_read(wav_file)
#     frame_len = int(frame_time * sample_rate)
#     step_len = int(step_time * sample_rate)
#     if pre_emphasis:
#         wav_signal = python_speech_features.sigproc.preemphasis(wav_signal)
#     framed_signal = frame_signal(wav_signal, frame_len, step_len, window_func)
#     frame_energy_list = np.zeros((framed_signal.shape[0],))
#     for frame_index in range(framed_signal.shape[0]):
#         frame_energy_list[frame_index] = np.sqrt(np.sum(framed_signal[frame_index] ** 2))
#     return frame_energy_list


class PBar(object):
    def __init__(self, max_value, progress_name: str = "Progress: "):
        _widgets_ = [progress_name, Percentage(), ' ', Bar('#'), ' ', Timer(), ' ', ETA(), ' ', FileTransferSpeed()]
        self.p_bar = ProgressBar(widgets=_widgets_, maxval=max_value)
        self.p_bar.start()

    def update(self, value):
        self.p_bar.update(value)

    def finish(self):
        self.p_bar.finish()


def fre2bark(fre_axis: Union[np.ndarray, float, None]) -> Union[np.ndarray, float, None]:
    if fre_axis is None:
        return None
    return 7.0 * np.arcsinh(fre_axis / 650.0)


def exception_printer(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        mute_third_party_logging()
        try:
            return function(*args, **kwargs)
        except KeyboardInterrupt as _err:
            raise _err
        except Exception as _err:
            logger.error("error")
            traceback.print_exc()

    return wrapper


def wait_for_jobs(jobs: list, executor: futures.ProcessPoolExecutor, progress_name: str = "Progress: ") -> List:
    try:
        p_bar = PBar(len(jobs), progress_name)
        complete_num = 0
        for _ in futures.as_completed(jobs):
            complete_num += 1
            p_bar.update(complete_num)

        results = []
        for job in jobs:
            results.append(job.result())
        p_bar.finish()

        return results
    except KeyboardInterrupt:
        logger.info("User terminated the process. Please wait for script kill all sub-processes......")
        executor.shutdown(False)

        process = psutil.Process(os.getpid())
        sub_process_list = process.children(True)
        for sub_process in sub_process_list:
            try:
                sub_process.terminate()
            except Exception:
                pass
        process.terminate()


def truncate_signal(wav_signal: np.ndarray, overflow_process_type: OverflowProcessType = OverflowProcessType.Scale, max_amp=32767, min_amp=-32768) -> np.ndarray:
    """
    clip the wav signal
    :return: clipped signal
    """
    if overflow_process_type == OverflowProcessType.Scale:
        weight = np.max(
            (1.0, np.max(wav_signal) / max_amp, np.min(wav_signal) / min_amp)
        )
        wav_signal = wav_signal / weight
    elif overflow_process_type == OverflowProcessType.Clip:
        wav_signal = np.clip(wav_signal, min_amp, max_amp)
    else:
        raise ValueError("The Value of overflow_process_type Error.")

    return wav_signal


def sqrt_hanning_window(length: int) -> np.ndarray:
    return np.sqrt(np.hanning(length))


def check_transaction(transaction: str, delete_stop_words: bool = False) -> str:
    # Convert case
    transaction = transaction.lower()

    # Process time
    transaction = transaction.replace("a.m.", "am")
    transaction = transaction.replace("p.m.", 'pm')
    transaction = transaction.replace(":00", "")

    # Filter non-English text and punctuation
    regex = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    transaction = re.sub(regex, " ", transaction)

    # Replace numbers
    number_dict = [('1', 'one'), ('2', 'two'), ('3', 'three'), ('4', 'four'), ('5', 'five'), ('6', "six"),
                   ('7', "seven"), ('8', 'eight'), ('9', "nine"), ('0', "zero")]
    for number_bin in number_dict:
        transaction = transaction.replace(number_bin[0], " " + number_bin[1] + " ")

    # Remove "-" connections, e.g. WI-FI
    transaction = transaction.replace('-', '')

    # Tokenize
    words_list = transaction.split()

    # Remove stop words
    if delete_stop_words:
        words_list = [w for w in words_list if w not in __stopwords__]

    # Stem extraction
    wnl = WordNetLemmatizer()
    words_list = [wnl.lemmatize(w) for w in words_list]

    return " ".join(words_list)


def reshape_single_dimension(raw_data: Union[int, float, Iterable, None], allow_none: bool = False, expected_length: int = MAX_FORMANT_NUM) -> Union[None, np.ndarray]:
    """
    reshape the single dimension parameter like "bandwidths" to shape (expected_length, )
    """
    if raw_data is None:
        if allow_none:
            return None
        else:
            raise ValueError('Raw Data Error.')

    if isinstance(raw_data, int) or isinstance(raw_data, float):
        raw_data = np.ones((expected_length,), dtype=np.float) * raw_data
        return raw_data
    elif isinstance(raw_data, Iterable):
        raw_data = np.array(raw_data)
        if len(raw_data.shape) == 1:
            if raw_data.shape[0] == 1:
                raw_data = np.tile(raw_data, (expected_length,))
            elif raw_data.shape[0] >= expected_length:
                raw_data = raw_data[:expected_length]
            else:
                raise ValueError('Raw Data Error.')

            return raw_data

    raise ValueError('Raw Data Error.')


def blue_noise(N: int, sample_rate: Union[int, float], min_fre: float, max_fre: float) -> np.ndarray:
    assert sample_rate / 2.0 >= max_fre > min_fre >= 0.0
    uneven = N % 2
    fft_length = N // 2 + 1 + uneven
    fft_freq = np.fft.fftfreq(N, 1.0 / sample_rate)[:fft_length]
    overflow_freq_index = np.argwhere(
        np.logical_or(fft_freq < min_fre, fft_freq > max_fre)
    )
    X = np.random.randn(fft_length) + 1j * np.random.randn(fft_length)
    X[overflow_freq_index] = 0
    S = np.sqrt(np.arange(fft_length))  # Filter
    y = (np.fft.irfft(X * S)).real
    if uneven:
        y = y[:-1]
    # y = y / np.max(np.abs(y))  # normalize the amplitude of the signal
    y = y / np.sqrt(np.sum(y ** 2))  # normalize the amplitude of the signal
    return y


def white_noise(N: int, sample_rate: Union[int, float], min_fre: float, max_fre: float) -> np.ndarray:
    assert sample_rate / 2.0 >= max_fre > min_fre >= 0.0
    uneven = N % 2
    fft_length = N // 2 + 1 + uneven
    fft_freq = np.abs(np.fft.fftfreq(N, 1.0 / sample_rate)[:fft_length])
    overflow_freq_index = np.argwhere(
        np.logical_or(fft_freq < min_fre, fft_freq > max_fre)
    )
    X = np.random.randn(fft_length) + 1j * np.random.randn(fft_length)
    X[overflow_freq_index] = 0
    y = (np.fft.irfft(X)).real
    if uneven:
        y = y[:-1]
    # y = y / np.max(np.abs(y))  # normalize the amplitude of the signal
    y = y / np.sqrt(np.sum(y ** 2))  # normalize the amplitude of the signal
    return y


def violet_noise(N: int, sample_rate: Union[int, float], min_fre: float, max_fre: float) -> np.ndarray:
    assert sample_rate / 2.0 >= max_fre > min_fre >= 0.0
    uneven = N % 2
    fft_length = N // 2 + 1 + uneven
    fft_freq = np.abs(np.fft.fftfreq(N, 1.0 / sample_rate)[:fft_length])
    overflow_freq_index = np.argwhere(
        np.logical_or(fft_freq < min_fre, fft_freq > max_fre)
    )
    X = np.random.randn(fft_length) + 1j * np.random.randn(fft_length)
    X[overflow_freq_index] = 0
    S = np.arange(fft_length)
    y = (np.fft.irfft(X * S)).real
    if uneven:
        y = y[:-1]
    # y = y / np.max(np.abs(y))  # normalize the amplitude of the signal
    y = y / np.sqrt(np.sum(y ** 2))  # normalize the amplitude of the signal
    return y


def pink_noise(N: int, sample_rate: Union[int, float], min_fre: float, max_fre: float) -> np.ndarray:
    assert sample_rate / 2.0 >= max_fre > min_fre >= 0.0
    uneven = N % 2
    fft_length = N // 2 + 1 + uneven
    fft_freq = np.abs(np.fft.fftfreq(N, 1.0 / sample_rate)[:fft_length])
    overflow_freq_index = np.argwhere(
        np.logical_or(fft_freq < min_fre, fft_freq > max_fre)
    )
    X = np.random.randn(fft_length) + 1j * np.random.randn(fft_length)
    X[overflow_freq_index] = 0
    S = np.sqrt(np.arange(fft_length) + 1.)
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    # y = y / np.max(np.abs(y))  # normalize the amplitude of the signal
    y = y / np.sqrt(np.sum(y ** 2))  # normalize the amplitude of the signal
    return y


def red_noise(N: int, sample_rate: Union[int, float], min_fre: float, max_fre: float) -> np.ndarray:
    assert sample_rate / 2.0 >= max_fre > min_fre >= 0.0
    uneven = N % 2
    fft_length = N // 2 + 1 + uneven
    fft_freq = np.abs(np.fft.fftfreq(N, 1.0 / sample_rate)[:fft_length])
    overflow_freq_index = np.argwhere(
        np.logical_or(fft_freq < min_fre, fft_freq > max_fre)
    )
    X = np.random.randn(fft_length) + 1j * np.random.randn(fft_length)
    X[overflow_freq_index] = 0
    S = np.arange(fft_length) + 1.
    y = (np.fft.irfft(X / S)).real
    if uneven:
        y = y[:-1]
    # y = y / np.max(np.abs(y))  # normalize the amplitude of the signal
    y = y / np.sqrt(np.sum(y ** 2))  # normalize the amplitude of the signal
    return y


class CustomEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.int) or isinstance(o, np.int32) or isinstance(o, np.int_) or isinstance(o, np.int16):
            return int(o)
        elif isinstance(o, np.float) or isinstance(o, np.float32) or isinstance(o, np.float16) or isinstance(o, np.float_):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, Enum):
            return o.name
        elif isinstance(o, Callable):
            return o.__name__
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return super(CustomEncoder, self).default(o)


def save_json_data(params: dict, output_path: str):
    with open(output_path, 'w') as _file_:
        json.dump(params, _file_, cls=CustomEncoder)


def get_dict_hash(dict_data: dict) -> str:
    return hashlib.md5(json.dumps(dict_data, cls=CustomEncoder).encode('utf-8')).hexdigest()[8:-8]


def numpy_snr(original_wav_path, adversarial_wav_path):
    _, original_signal = wav_read(original_wav_path)
    _, adversarial_signal = wav_read(adversarial_wav_path)
    signal_sum = np.sum(original_signal ** 2)
    noise_sum = np.sum((original_signal - adversarial_signal) ** 2)
    snr = 10 * np.log10(signal_sum / noise_sum)
    return snr


def feature_normalize(feature: Union[list, np.ndarray]) -> np.ndarray:
    mu = np.mean(feature)
    sigma = np.std(feature)
    return (feature - mu) / sigma


def filter_irrelevant_wav(wav_files: List[str]) -> List[str]:
    for wav_suffix in IRRELEVANT_WAV_SUFFIX:
        wav_files = [wav_file for wav_file in wav_files if not wav_file.endswith(wav_suffix)]
    return wav_files


def get_wav_hash(wav_file: str) -> str:
    with open(wav_file, 'rb') as _file_:
        hashObject = hashlib.sha512()
        hashObject.update(_file_.read())
        file_hash = hashObject.hexdigest()
    return file_hash

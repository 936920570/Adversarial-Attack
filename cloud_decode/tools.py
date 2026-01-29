# import matplotlib.pyplot as plt
# import torch, torchaudio
import os, datetime, glob, json
# import numpy as np
from argparse import ArgumentParser
from types import SimpleNamespace
from configparser import ConfigParser
from ast import literal_eval
from PIL import Image
# from torchvision import transforms, utils


_log_file_path = os.path.join(os.path.expanduser('~'), 'log.txt')


def dict_eval(d):
    for k, v in d.items():
        try:  # number/tuple/dict/list type
            d[k] = literal_eval(v)
        except:  # str type
            pass
    return d


def parse_config(path):
    cfg = ConfigParser()
    cfg.read(path, encoding='utf-8')
    config = {}
    for section in cfg.sections():
        config[section] = SimpleNamespace(**dict_eval(dict(cfg[section])))
    return SimpleNamespace(**config)


# def label_to_id(label):
#     return list(_id_and_labels.keys())[list(
#         _id_and_labels.values()).index(label)]


# def id_to_label(id):
#     return _id_and_labels[id]


# def show_image(path: str, title: str = ''):
#     img = Image.open(path).convert('RGB')
#     plt.imshow(np.array(img))
#     plt.title(title)
#     plt.show()


# def show_image_tensor(image_tensor: torch.Tensor, title: str = ''):
#     plt.figure()
#     if image_tensor.dim() == 4:
#         plt.imshow(torch.permute(image_tensor[0], (1, 2, 0)).numpy())
#     elif image_tensor.dim() == 3:
#         plt.imshow(image_tensor[0].numpy(), cmap='gray')
#     elif image_tensor.dim() == 2:
#         plt.imshow(image_tensor.numpy(), cmap='gray')
#     else:
#         raise ValueError('Unparsable tensor dimension:', image_tensor.dim())
#     plt.title(title)
#     plt.show()


def load_image(path):
    return Image.open(path).convert('RGB')


# def load_image_tensor(path: str,
#                       size: tuple = None,
#                       norm: tuple = None) -> torch.Tensor:
#     img = load_image(path)
#     if size is not None:
#         img = transforms.Resize(size)(img)
#     img_tensor = transforms.ToTensor()(img)
#     if norm is not None:
#         img_tensor = transforms.Normalize(norm[0], norm[1])(img_tensor)
#     return img_tensor[None, :, :, :]


# def predict(model: torch.nn.Module, x: torch.Tensor, imagenet: bool = True):
#     with torch.no_grad():
#         outputs = model(x)
#         id_ = torch.argmax(outputs.view(-1)).item()
#         if imagenet is True:
#             return id_to_label(id_)
#         else:
#             return id_


def log_to_file(*logs: list, p: bool = True):
    log = ' '.join([str(item) for item in logs])
    log_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + log
    with open(_log_file_path, 'a') as log_file:
        log_file.write(log_str + '\n')
    if p is True:
        print(log_str)


# def get_sizes_and_plot(img_dir: str):
#     import seaborn as sns
#     sizes = []
#     for root, _, imgs in os.walk(img_dir):
#         for img in imgs:
#             sizes.append(Image.open(os.path.join(root, img)).size)
#     sns.jointplot(data=sizes)
#     plt.show()
#     return sizes


def print_layer_names(model_name: str):
    import timm
    m = timm.create_model(model_name)
    for n, _ in m.named_modules():
        print(n)


# def power_db(signal):
#     p = torch.sum(torch.square(signal))
#     return (10 * torch.log10(p)).item()


# def SNR(src, adv):
#     noise_p = torch.sum(torch.square(src - adv))
#     signal_p = torch.sum(torch.square(src))
#     return (10 * torch.log10(signal_p / noise_p)).item()


# def load_wav(path, sr=16000):
#     torchaudio.set_audio_backend('soundfile')
#     wave, sr_ori = torchaudio.load(path)
#     if sr_ori != sr:
#         wave = torchaudio.functional.resample(wave, sr_ori, sr, lowpass_filter_width=64)
#     return wave


# def load_wav2(path1, path2, sr=16000):
#     wave1 = load_wav(path1, sr)
#     wave2 = load_wav(path2, sr)
#     ori_shape = (wave1.shape[1], wave2.shape[1])
#     n_samples = min(wave1.shape[1], wave2.shape[1])  # ! cut the longer one
#     wave1, wave2 = wave1[:, :n_samples], wave2[:, :n_samples]
#     return wave1, wave2, ori_shape


def makedir(path):
    dirname = os.path.dirname(path)
    if os.path.exists(dirname) is False:
        os.makedirs(dirname)
    return path


# def save_wav(wave, path, sr=16000):
#     torchaudio.save(makedir(path),
#                     wave,
#                     sample_rate=sr,
#                     format='wav',
#                     encoding='PCM_S',
#                     bits_per_sample=16)
#     return path


def find_all_ext(root, ext: str):
    paths = []
    for dir_name, _, _ in os.walk(root):
        paths += glob.glob(os.path.join(dir_name, '*.' + ext))
    return paths


def find_file(root, pattern: str) -> str:
    for dir_name, _, _ in os.walk(root):
        paths = glob.glob(os.path.join(dir_name, pattern))
        if len(paths) != 0:
            return paths[0]  # ! only return the first one


def read_file(path, **kwargs) -> str:
    with open(path, **kwargs) as f:
        return f.read().strip()


def write_file(path, text, **kwargs):
    with open(makedir(path), 'w', **kwargs) as f:
        f.write(text)
        return text


def read_json(path, **kwargs):
    with open(path, **kwargs) as f:
        return json.load(f)


def write_json(path, data: dict, update=False, **kwargs):
    data_w = data
    if update is True:
        info: dict = read_json(path)
        data_w = info.update(data_w)
    with open(makedir(path), 'w', **kwargs) as f:
        f.write(json.dumps(data_w, indent=1))


# def get_1st_conv_layer(m):
#     for n, l in m.named_modules():
#         if isinstance(l, torch.nn.Conv2d) or isinstance(l, torch.nn.Conv1d):
#             return n, l


def set_layer(m, layer_name: str, new_layer):
    names = layer_name.split('.')
    parent = m
    for name in names[:-1]:
        parent = getattr(parent, name)
    setattr(parent, names[-1], new_layer)


# def unit_mat(size: tuple):
#     return torch.eye(np.prod(size)).float()


def filename(path: str):
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--function', type=str)
    parser.add_argument('--args_list', type=str)
    args = parser.parse_args()
    f_args = literal_eval(args.args_list)
    globals()[args.function](*f_args)

import torch
import torch.nn.functional as F
import torch.optim as optim
from warnings import simplefilter
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import scipy
import librosa
import random
import imageio
import wave
from typing import Union
import os
import glob
import math 
import json
import matplotlib.pyplot as plt
import sys
from cmaes import CMA
from cloud_decode.aliyun_function import *
from cloud_decode.tencentyun_function import *
from cloud_decode.google_api import *
from cloud_decode.azure_api import *
from cloud_decode.openai_function import openai_recong

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
if device.type == 'cuda':
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')

simplefilter(action='ignore', category=Warning)
parser = argparse.ArgumentParser(description='ASR attack')
plt.rcParams.update({'font.size': 24})
torch.backends.cudnn.enabled = False

parser.add_argument('--seed', type=int, default=2025, metavar='S',help='random seed (default: 2023)')
parser.add_argument('--epoch', type=int, default=1500)
parser.add_argument('--speech-file-path', default='',type=str, help='speech file path')
parser.add_argument('--music-file-path', default='',type=str, help='music file path')
parser.add_argument('--attack-target', default='aliyun',type=str,choices=['tencentyun','aliyun','baiduyun','iflytec','google','azure','openai'])
parser.add_argument('--sample-num', default=0,type=int, help='samples num')
parser.add_argument('--sound-db', default=75,type=int, help='sound db')
parser.add_argument('--start-second', type=float, default=0.0, help='start second in music to add perturbation')

class Attacker:
    def __init__(self, args):
        self.args = args
        self.epoch=args.epoch
        self.sound_db = args.sound_db
        self.start_second = args.start_second
        self.n_fft=512
        self.hop_length=256
        self.frame_num=240
        self.seq_len=self.hop_length*(self.frame_num-1)+self.n_fft

        if args.attack_target == 'tencentyun':
            self.decode_function=tencent_recong
        if args.attack_target == 'aliyun':
            self.decode_function=aliyun_recong
        if args.attack_target == 'baiduyun':
            self.decode_function=google_decode
        if args.attack_target == 'azure':
            self.decode_function=azure_decode
        if args.attack_target == 'openai':
            self.decode_function=openai_recong

        self.normalize=normalize_e
        self.denormalize=denormalize_e
        # Load command file
        self.speech,_=self.read_wav_file(args.speech_file_path)
        speech_name=args.speech_file_path.split('/')[-1].split('.')[0]
        self.command=speech_name.replace('_',' ')
        physical_samples_path=Path('physical_samples')/speech_name/args.attack_target
        if not os.path.exists(physical_samples_path):
            os.makedirs(physical_samples_path)
        self.physical_samples_path=physical_samples_path

        self.decode_count=0
        music_file_list=glob.glob(args.music_file_path+'/*.wav')
        print(f"Found {len(music_file_list)} music files to attack.")

        for music_file in music_file_list:
            # Load music file
            music_name=music_file.split('/')[-1].split('.')[0]
            music,start_time=self.read_wav_file(music_file)
            self.attack_music_name=speech_name+'_'+music_name+'_'+str(start_time)
            print(self.attack_music_name,'  attack begin!!')
            success=self.distribution_attack(self.speech,music)
            if success:
                print(self.attack_music_name,'  Success!!')
            else:
                print(self.attack_music_name,'  Failed!!')
                

    def read_wav_file(self,file_path,start_time=0):
        wav_data,_ = librosa.load(file_path, sr=16000, mono=True)
        if len(wav_data)>self.seq_len:
            if not start_time:
                start_time=0
            wav_data=wav_data[start_time:start_time+self.seq_len]
        else:
            wav_data=wav_data[np.convolve(np.abs(wav_data), np.ones((512))/512, mode='same')>0.001]
        mean_amp = np.mean(np.abs(wav_data*(2**15)))
        scale_ratio = self.compute_scale_ratio(mean_amp, self.sound_db, max_amp_value=32768.)
        wav_data = scale_ratio * wav_data
        wav_data=np.clip(np.array(wav_data).ravel(),-1,1)
        return wav_data,start_time

    def compute_scale_ratio(self,amp, expected_db, max_amp_value: float = 32768.):
        return np.power(10, (expected_db + 20 * np.log10(max_amp_value) - 96.) / 20) / amp
    
    
    def distribution_attack(self,speech,music): 
        music_mag,music_stft_phase=wav_preprocess(music,self.n_fft,self.hop_length)
        music_mag=self.normalize(music_mag)
        clean_music=conc_tog_specphase(music_mag.unsqueeze(0),music_stft_phase.unsqueeze(0),n_fft=self.n_fft,hop_length=self.hop_length).squeeze()
        speech_mag,_=wav_preprocess(speech,self.n_fft,self.hop_length)
        speech_mag=self.normalize(speech_mag)
        speech_frame=speech_mag.size(1)
        speech_rhythm=F.softmax(torch.mean(square_smooth(self.denormalize(speech_mag),square_kernel_size=[7]),dim=0))


        smooth_music_mag=square_smooth(music_mag,kernel_size=[15])
        seq_idx = 0
        frame_idx = int(self.start_second * 16000 / self.hop_length)
        
        # Ensure frame_idx does not exceed total music frames
        max_frame_idx = music_mag.size(1) - speech_mag.size(1)
        if frame_idx > max_frame_idx:
            print(f"Warning: start_second={self.start_second}s exceeds music length, using max possible frame_idx={max_frame_idx}")
            frame_idx = max(0, max_frame_idx)
        
        #print('frame_idx:', frame_idx, '    speech_frame:', speech_frame.item())

        smooth_speech_mag=square_smooth(speech_mag,kernel_size=[15])   
        speech_feature=(smooth_speech_mag-square_smooth(smooth_speech_mag,kernel_size=[15]))
        music_feature=(smooth_music_mag-square_smooth(smooth_music_mag,kernel_size=[15]))[:,frame_idx:frame_idx+speech_frame]
        feature_mask=square_smooth(torch.tensor(square_smooth(torch.abs(speech_feature),kernel_size=[7])<0.005).float(),kernel_size=[5])
        #target_feature=speech_feature+feature_mask*music_feature

                # Create frequency mask (200Hz - 7000Hz)
        sr = 16000
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        freq_mask = (freqs >= 200) & (freqs <= 7000)
        # Convert to Tensor and adjust shape for broadcasting
        self.freq_mask_tensor = torch.tensor(freq_mask, dtype=torch.float32, device=device).unsqueeze(1)

        advised_gain=torch.zeros_like(music_mag)
        #advised_gain = 1e-3 * torch.rand_like(music_mag);
        advised_gain.requires_grad=True
        optimizer=optim.Adam(params=[advised_gain],lr=4e-2)
        balance_param=30
        with tqdm(total=self.epoch, desc='Adversarial Train') as train_enum:
            for epoch in range(self.epoch):
                with torch.autograd.set_detect_anomaly(True):
                    # Smooth perturbation and enforce non-negative (use softplus to maintain differentiability), ensuring perturbation is always greater than zero
                    raw_smooth_advised = square_smooth(advised_gain, square_kernel_size=[3])
                    # Use softplus to map arbitrary real numbers to positive values, avoiding negative perturbations
                    smooth_advised_gain = F.softplus(raw_smooth_advised) * self.freq_mask_tensor
                    #smooth_advised_gain = F.softplus(raw_smooth_advised)
                    advised_mag=torch.clamp_min(music_mag+smooth_advised_gain,0)
                    
                    # Calculate audio segment at corresponding position
                    start_sample = frame_idx * self.hop_length
                    end_sample = (frame_idx + speech_frame) * self.hop_length + self.n_fft
                    full_advised_audio = conc_tog_specphase(advised_mag.unsqueeze(0), music_stft_phase.unsqueeze(0), 
                                                               n_fft=self.n_fft, hop_length=self.hop_length).squeeze()
                        
                    advised_audio = full_advised_audio[start_sample:end_sample]
                    original_music = clean_music[start_sample:end_sample]
                    
                    # Use command audio superimposed with original music at corresponding position as target_audio
                    speech_tensor = torch.tensor(speech, dtype=torch.float32).to(device)
                    
                    # Ensure consistent length (take the shorter length)
                    min_length = min(speech_tensor.shape[0], original_music.shape[0], advised_audio.shape[0])
                    speech_tensor = speech_tensor[:min_length]
                    original_music = original_music[:min_length]
                    advised_audio = advised_audio[:min_length]
                    
                    # Target = command + corresponding part of original music
                    target_audio = speech_tensor

                    # Calculate 80-dimensional log-Mel spectrogram (apply CMVN normalization)
                    target_log_mel = compute_log_mel_spectrogram(target_audio, n_fft=self.n_fft, 
                                                                    hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=True)
                    advised_log_mel = compute_log_mel_spectrogram(advised_audio, n_fft=self.n_fft, 
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=True)
                    advised_mel=compute_log_mel_spectrogram(advised_audio, n_fft=self.n_fft,
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)
                    original_music_log_mel = compute_log_mel_spectrogram(original_music, n_fft=self.n_fft,
                                                                            hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=True)
                    original_music_mel=compute_log_mel_spectrogram(original_music, n_fft=self.n_fft,
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)
                    # Check for NaN or Inf
                    if not torch.isfinite(target_log_mel).all() or not torch.isfinite(advised_log_mel).all():
                        print("Warning: NaN or Inf detected in log-mel spectrograms, skipping this iteration")
                        continue
                    
                    # Ensure both log-Mel spectrograms have consistent time dimensions
                    min_time_frames = min(target_log_mel.shape[1], advised_log_mel.shape[1])
                    target_log_mel = target_log_mel[:, :min_time_frames]
                    advised_log_mel = advised_log_mel[:, :min_time_frames]
                    original_music_log_mel = original_music_log_mel[:, :min_time_frames]
                    advised_mel = advised_mel[:, :min_time_frames]
                    original_music_mel = original_music_mel[:, :min_time_frames]
                    # Calculate MAE difference for entire log-Mel spectrogram
                    feature_loss = F.l1_loss(advised_log_mel, target_log_mel)

                    # Calculate advised_log_mel / advised_mel directly from spectrum amplitude (no need to convert to audio first)
                    seg_len = int(speech_frame.item()) if torch.is_tensor(speech_frame) else int(speech_frame)
                    # Calculate 80-dimensional log-Mel spectrogram (apply CMVN normalization) - only for noise loss calculation
                    advised_mel=compute_log_mel_spectrogram(advised_audio, n_fft=self.n_fft,
                                                                hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)
                    
                    original_music_mel=compute_log_mel_spectrogram(original_music, n_fft=self.n_fft,
                                                                 hop_length=self.hop_length, n_mels=80, sr=16000, apply_cmvn_norm=False)

                
                    seg_len = int(speech_frame.item()) if torch.is_tensor(speech_frame) else int(speech_frame)
                    seg_start = frame_idx
                    seg_end = frame_idx + seg_len
                    T_time = music_mag.shape[1]
                    if seg_start >= T_time:
                        continue
                    seg_end = min(seg_end, T_time)
                
                    # Check if feature_loss is finite
                    if not torch.isfinite(feature_loss):
                        print("Warning: feature_loss is not finite, using zero loss")
                        feature_loss = torch.tensor(0.0, device=device, requires_grad=True)
                    
                    # frame_snrs = []
                    # for frame_idx in range(advised_mel.shape[1]):  # Iterate over each time frame
                    #     # Extract signal and noise for current frame
                    #     frame_signal = original_music_mel[:, frame_idx]  # shape: [n_mels]
                    #     frame_noise = advised_mel[:, frame_idx] - frame_signal
                    #     # Calculate SNR for current frame
                    #     frame_snr = torch.sum(frame_noise ** 2) / (torch.sum(frame_signal ** 2) + 1e-8)
                    #     if torch.isfinite(frame_snr):
                    #         frame_snrs.append(frame_snr)
                    # noise_loss = torch.mean(torch.stack(frame_snrs))
                    #loss=feature_loss+40*noise_loss
                    #Close to music? loss=feature_loss+0.4*F.l1_loss(advised_log_mel,original_music_log_mel)
                    #loss=feature_loss+0.30*torch.norm(advised_mel-original_music_mel,p=2)
                    #loss = diff_loss_freq+F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))
                    #loss = feature_loss+87.5*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))
                    loss = feature_loss+50*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))
                    #noise_loss=F.mse_loss(advised_mel/ (original_music_mel + 1e-8),torch.ones_like(original_music_mel))
                    #loss=feature_loss+0.7*noise_loss#0.8 whisper
                    #noise_loss=F.l1_loss(advised_log_mel/ (original_music_log_mel + 1e-8),torch.ones_like(original_music_log_mel))
                    #√ loss=feature_loss+0.4*noise_loss
                    #loss=feature_loss+30*noise_loss
                    #loss = feature_loss+400*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))+10*noise_loss
                    #loss = feature_loss+200*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))+0.01*torch.norm(advised_mel-original_music_mel,p=2)
                    #loss = feature_loss+600*F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))+5*noise_loss
                    # Ensure loss is finite
                    if not torch.isfinite(loss):
                        print("Warning: total loss is not finite, skipping backward pass")
                        continue
                        
                    
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    
                    # Check gradients
                    if torch.isfinite(advised_gain.grad).all():
                        optimizer.step()
                    else:
                        print("Warning: gradients contain NaN or Inf, skipping optimizer step")
                        
                    train_enum.set_description(f'Train attacker:(loss:{loss:.10f},feature_loss:{feature_loss:.8f})')
                    train_enum.update()

                #torch.cat((interpolate_smooth_speech_mag,speech_feature,feature_mask,advised_mag,music_mag),1).cpu().detach().numpy()
        mask=torch.zeros_like(music_mag)
        mask[5:,frame_idx-2:frame_idx+speech_frame+2]=1
        mask=square_smooth(mask,square_kernel_size=[5])
        feature_mask=square_smooth(torch.tensor(square_smooth(torch.abs(speech_feature),kernel_size=[7])>0.001).float(),kernel_size=[5])
        mask[:,frame_idx:frame_idx+speech_frame]*=feature_mask
        advised_mag=torch.clamp_min(music_mag+(smooth_advised_gain)*mask,0)
        #
        #imageio.imwrite('attack_stft.png',torch.cat((interpolate_smooth_speech_mag,speech_feature,feature_mask,advised_mag,music_mag),1).cpu().detach().numpy())
        wav=GLA(self.denormalize(advised_mag).unsqueeze(0),music_stft_phase.unsqueeze(0),self.n_fft,self.hop_length).squeeze()
            
        noise=wav-clean_music
        SNR=10*torch.log10((torch.sum(clean_music[:]**2))/(torch.sum(noise**2)))
        audio_file = f"{self.attack_music_name}_{self.args.attack_target}_SNR{SNR:.3f}.wav"
        wav_write(np.clip(wav.cpu().detach().numpy().ravel(),-0.9,0.9),audio_file,16000)
        result,success = self.decode_function(audio_file)
        self.decode_count+=1
        assert success,'Decode Failed!!!'
        print("recognition answer:"+result)
        print(self.physical_samples_path)
        #time.sleep(20)
        if self.command not in result.lower().replace('\'s',' is').replace(',','').replace('.','').replace('?','').replace('!','').replace('9','nine').replace('1','one'):
            return False
        else:

            clean_music_np = np.clip(clean_music.cpu().detach().numpy().ravel(),-1,1)
            wav_np = np.clip(wav.cpu().detach().numpy().ravel(),-1,1)
            
            # Calculate weighted segment SNR
            noise_np = wav_np - clean_music_np
            seg_snr = self.calculate_seg_snr(clean_music_np, noise_np, sr=16000, segment_ms=50)

            # Add ViSQOL score and segSNR to filename
            new_audio_filename = f"{self.attack_music_name}_{self.args.attack_target}_SNR{SNR:.3f}_segSNR{seg_snr:.3f}.wav"
            # Use os.path.join and normpath to handle separators correctly on Windows
            success_audio_file = os.path.normpath(os.path.join(str(self.physical_samples_path), new_audio_filename))
            
            # Save adversarial sample
            wav_write(wav_np,success_audio_file,16000)
            # with open(success_audio_file.replace('.wav','.json'), 'w') as json_file:
            #     json.dump({'result':result}, json_file, ensure_ascii=False)
            self.decode_count=0
            # CMA-ES optimization
            variable_num=((speech_frame+20)//10)*int(self.n_fft/20)
            optimizer = CMA(mean=np.ones(variable_num),sigma=0.05,bounds=np.array([[0,1]]*variable_num),population_size=15)
            best_value=1e11
            best_generation=0
            start_index=(frame_idx-5)*self.hop_length
            noise_len=(speech_frame+10)*self.hop_length+self.n_fft
            for generation in range(1000):
                solutions = []
                for _ in range(optimizer.population_size):
                    variable = optimizer.ask()
                    value=0
                    CMA_advised_mag=smooth_advised_gain.clone()
                    CMA_advised_mag[:,frame_idx-10:frame_idx+speech_frame+10]*=square_smooth(F.interpolate(torch.tensor(variable).view(1,1,int(self.n_fft/20),-1).to(device),size=(int(self.n_fft/2+1), speech_frame+20), mode='area').squeeze().float(),square_kernel_size=[9])
                    target_mag=denormalize_e(torch.clamp_min(music_mag+CMA_advised_mag,0))
                    cmaes_wav=GLA(target_mag.unsqueeze(0),music_stft_phase.unsqueeze(0),self.n_fft,self.hop_length).squeeze()
                    
                    sf = speech_frame.item() if torch.is_tensor(speech_frame) else speech_frame
                    start_sample = (frame_idx - 5) * self.hop_length
                    end_sample = int((frame_idx + sf + 5) * self.hop_length + self.n_fft)
                    time_mask = torch.zeros_like(wav)
                    time_mask[start_sample:end_sample] = 1.0
                    cmaes_wav = clean_music + (cmaes_wav - clean_music) * time_mask
                    
                    wav_write(np.clip(cmaes_wav.cpu().detach().numpy().ravel(),-1,1),audio_file,16000)
                    #SNR_star=10*torch.log10((torch.sum(clean_music[start_index:start_index+noise_len]**2))/(torch.sum(noise**2)))
                    cmaes_result,success= self.decode_function(audio_file)
                    #print("recognition answer:"+cmaes_result)
                    assert success,'Decode Failed!!!'
                    value+=torch.sum(torch.abs(target_mag-denormalize_e(music_mag))).item()
                    if self.command not in cmaes_result.lower().replace('\'s',' is').replace(',','').replace('.','').replace('?','').replace('!','').replace('9','nine').replace('1','one'):
                        value=5e10
                    if value < best_value and value<1e10:
                        best_value=value
                        best_generation=generation
                        best_cmaes_wav=cmaes_wav
                        #imageio.imwrite('attack_stft.png',torch.cat((smooth_speech_mag,torch.clamp_min(normalize_e(target_mag),0)),1).cpu().detach().numpy())
                        wav_write(np.clip(best_cmaes_wav.cpu().detach().numpy().ravel(),-1,1),success_audio_file,16000)

                    solutions.append((variable,value))
                    print(f'#{generation}: {value}  target:{self.args.attack_target}  SNR:{SNR:.5f}')
                if generation-best_generation>5:
                    break
                optimizer.tell(solutions)
            if best_value<1e10:
                # Calculate final SNR and SegSNR
                best_clean_music_np = np.clip(clean_music.cpu().detach().numpy().ravel(),-1,1)
                best_wav_np = np.clip(best_cmaes_wav.cpu().detach().numpy().ravel(),-1,1)
                best_noise_np = best_wav_np - best_clean_music_np
                best_SNR = 10*np.log10((np.sum(best_clean_music_np[:]**2))/(np.sum(best_noise_np**2)))
                best_seg_snr = self.calculate_seg_snr(best_clean_music_np, best_noise_np, sr=16000, segment_ms=50)
                
                # Use self.attack_music_name for renaming as well
                base_filename = f"{self.attack_music_name}_{self.args.attack_target}_SNR{best_SNR:.3f}_segSNR{best_seg_snr:.3f}.wav"
                save_file = os.path.normpath(os.path.join(str(self.physical_samples_path), base_filename))
                
                if os.path.exists(success_audio_file):
                    os.rename(success_audio_file, save_file)
                else:
                    print(f"[Error] Could not rename file. Source file not found: {success_audio_file}")

                # Also rename corresponding clean_music file
                # clean_music_file_old = success_audio_file.replace('.wav', '_clean.wav')
                # clean_music_file_new = save_file.replace('.wav', '_clean.wav')
                # if os.path.exists(clean_music_file_old):
                #     os.rename(clean_music_file_old, clean_music_file_new)
                #os.system('rm -f '+audio_file)
            
                print(f'cma_es success!!! best_value:{best_value}')
            else:
                print('cma_es Failed!!!')

            return True


    def calculate_seg_snr(self, clean_audio, noise_audio, sr=16000, segment_ms=50):
        """
        Calculate weighted segmental SNR (Weighted Segmental SNR)
        Args:
            clean_audio: Clean audio (numpy array)
            noise_audio: Noise audio (numpy array)
            sr: Sample rate
            segment_ms: Segment length (milliseconds)
        Returns:
            weighted_seg_snr: Weighted segmental SNR value
        """
        # Ensure consistent length
        min_len = min(len(clean_audio), len(noise_audio))
        clean = clean_audio[:min_len]
        noise = noise_audio[:min_len]
        
        # Calculate samples per segment
        seg_len = int(sr * segment_ms / 1000)
        num_segments = min_len // seg_len
        
        if num_segments == 0:
            return 0.0
            
        segment_snrs = []
        segment_noise_energies = []
        
        for i in range(num_segments):
            start = i * seg_len
            end = start + seg_len
            
            seg_clean = clean[start:end]
            seg_noise = noise[start:end]
            
            clean_energy = np.sum(seg_clean ** 2)
            noise_energy = np.sum(seg_noise ** 2)
            
            # Avoid division by zero and log(0)
            if noise_energy < 1e-10:
                snr = 100.0 # Noise is extremely small, SNR is very high
                continue
            elif clean_energy < 1e-10:
                snr = -100.0 # Signal is extremely small, SNR is very low
            else:
                snr = 10 * np.log10(clean_energy / noise_energy)
                
            segment_snrs.append(snr)
            segment_noise_energies.append(noise_energy)
            
        # Calculate weighted average
        total_noise_energy = sum(segment_noise_energies)
        if total_noise_energy < 1e-10:
            return np.mean(segment_snrs)
            
        weights = [e / total_noise_energy for e in segment_noise_energies]
        weighted_seg_snr = sum(w * s for w, s in zip(weights, segment_snrs))
        
        return weighted_seg_snr


def apply_cmvn(log_mel, eps=1e-8):
    """
    Apply CMVN (Cepstral Mean and Variance Normalization) to log-Mel spectrogram
    Args:
        log_mel: log-Mel spectrogram tensor, shape [n_mels, n_frames]
        eps: Small value to prevent division by zero
    Returns:
        normalized_log_mel: CMVN normalized log-Mel spectrogram
    """
    # Calculate mean and variance for each Mel band (along time dimension)
    mean = torch.mean(log_mel, dim=1, keepdim=True)  # shape: [n_mels, 1]
    var = torch.var(log_mel, dim=1, keepdim=True, unbiased=False)  # shape: [n_mels, 1]
    
    # CMVN normalization: (x - mean) / sqrt(var + eps)
    normalized_log_mel = (log_mel - mean) / torch.sqrt(var + eps)
    
    return normalized_log_mel


def compute_mfcc_features(audio_tensor, n_fft=512, hop_length=258, n_mels=80, n_mfcc=13, sr=16000):
    """
    Calculate MFCC features of audio
    Args:
        audio_tensor: Input audio tensor
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of Mel filters
        n_mfcc: Number of MFCC coefficients
        sr: Sample rate
    Returns:
        mfcc: MFCC features tensor, shape [n_mfcc, n_frames]
    """
    # Keep tensor format to maintain gradients
    if isinstance(audio_tensor, torch.Tensor):
        audio_for_mfcc = audio_tensor
    else:
        audio_for_mfcc = torch.tensor(audio_tensor, dtype=torch.float32).to(device)
    
    # Use torch.stft to calculate spectrogram, maintaining gradients
    stft = torch.stft(
        audio_for_mfcc, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(device),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Calculate power spectrum
    power_spec = torch.abs(stft) ** 2
    
    # Create Mel filter bank
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr/2)
    mel_filters = torch.tensor(mel_filters, dtype=torch.float32).to(device)
    
    # Apply Mel filters
    mel_spec = torch.matmul(mel_filters, power_spec)
    
    # Ensure all values in mel_spec are positive, avoiding log(0) or log(negative)
    mel_spec = torch.clamp_min(mel_spec, 1e-10)
    
    # Convert to log scale
    log_mel = torch.log(mel_spec)
    
    # Check and handle possible NaN or Inf values
    log_mel = torch.where(torch.isfinite(log_mel), log_mel, torch.zeros_like(log_mel))
    
    # Calculate DCT (Discrete Cosine Transform) to get MFCC
    # Create DCT matrix
    n_frames = log_mel.shape[1]
    dct_matrix = torch.zeros(n_mfcc, n_mels, dtype=torch.float32).to(device)
    
    for i in range(n_mfcc):
        for j in range(n_mels):
            dct_matrix[i, j] = np.cos(np.pi * i * (j + 0.5) / n_mels)
        if i == 0:
            dct_matrix[i, :] *= np.sqrt(1.0 / n_mels)
        else:
            dct_matrix[i, :] *= np.sqrt(2.0 / n_mels)
    
    # Apply DCT: MFCC = DCT_matrix @ log_mel
    mfcc = torch.matmul(dct_matrix, log_mel)
    
    return mfcc


def compute_log_mel_spectrogram(audio_tensor, n_fft=512, hop_length=258, n_mels=80, sr=16000, apply_cmvn_norm=True):
    """
    Calculate log-Mel spectrogram of audio
    Args:
        audio_tensor: Input audio tensor
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of Mel filters
        sr: Sample rate
        apply_cmvn_norm: Whether to apply CMVN normalization
    Returns:
        log_mel: log-Mel spectrogram tensor
    """
    # Keep tensor format to maintain gradients
    if isinstance(audio_tensor, torch.Tensor):
        audio_for_mel = audio_tensor
    else:
        audio_for_mel = torch.tensor(audio_tensor, dtype=torch.float32).to(device)
    
    # Use torch.stft to calculate spectrogram, maintaining gradients
    stft = torch.stft(
        audio_for_mel, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=n_fft,
        window=torch.hann_window(n_fft).to(device),
        center=True,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Calculate power spectrum
    power_spec = torch.abs(stft) ** 2
    
    # Create Mel filter bank
    mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmax=sr/2)
    mel_filters = torch.tensor(mel_filters, dtype=torch.float32).to(device)
    
    # Apply Mel filters
    mel_spec = torch.matmul(mel_filters, power_spec)
    
    # Ensure all values in mel_spec are positive, avoiding log(0) or log(negative)
    mel_spec = torch.clamp_min(mel_spec, 1e-10)
    if not apply_cmvn_norm:
        return mel_spec
    # Convert to log scale, using torch.clamp for numerical stability
    log_mel = torch.log(mel_spec)
    
    # Check and handle possible NaN or Inf values
    log_mel = torch.where(torch.isfinite(log_mel), log_mel, torch.zeros_like(log_mel))
    
    # Apply CMVN normalization
    #log_mel = apply_cmvn(log_mel)

    return log_mel


def square_smooth(input,square_kernel_size=[],kernel_size=[]):
    bias=torch.zeros(1).to(device)
    input=input.clone().unsqueeze(0).unsqueeze(0)
    for size in square_kernel_size:
        kernel=(torch.ones(1,1,size,size)/size**2).to(device)
        padding_size=int((size-1)/2)
        input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='replicate')
        input=F.conv2d(input, kernel, bias, stride=1)
    for size in kernel_size:
        kernel=(torch.ones(1,1,size,1)/size).to(device)
        padding_size=int((size-1)/2)
        input=F.pad(input,(0,0,padding_size,padding_size),mode='replicate')
        input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()


def edge_detection(input):
    bias=torch.zeros(1).to(device)
    input=input.clone().unsqueeze(0).unsqueeze(0)
    kernel=torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float).to(device)
    padding_size=1
    input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='replicate')
    input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()

def standardization(stft_distribution):
    return (stft_distribution-stft_distribution.mean())/stft_distribution.std()
    

def wav_write(wav_signal: np.ndarray, wav_path: str, sample_rate: Union[int, float], scale_bit_length: bool = True, bit_length: int = 16):
    """
    write the wav file
    :param wav_signal: wav signal.
    :param wav_path: wav path.
    :param scale_bit_length: whether scale signal to bit length.
    :param sample_rate: sample rate.
    :param bit_length: bit length.
    """
    output_dir = os.path.dirname(wav_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    wav_handler = wave.open(wav_path, "wb")
    wav_handler.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
    max_amp = 2 ** (bit_length - 1)
    if scale_bit_length:
        wav_data = (wav_signal * max_amp).astype(np.int16)
    else:
        wav_data = wav_signal.astype(np.int16)
    wav_data = np.clip(wav_data, -max_amp, max_amp - 1)
    wav_handler.writeframes(wav_data.tobytes())
    wav_handler.close()

def normalize_e(stft_mag):
    return torch.log(stft_mag+1)

def denormalize_e(stft_mag):
    return torch.pow(math.e,stft_mag)-1

def normalize_10(stft_mag):
    return (torch.log10(torch.clamp_min(stft_mag,1e-10))+2.5)/7.5

def denormalize_10(stft_mag):
    return torch.pow(10,stft_mag*7.5-2.5)

def conc_tog_specphase(S, P,n_fft,hop_length):
    S = denormalize_e(S)
    P = P * np.pi
    SP = S * torch.complex(torch.cos(P),torch.sin(P))
    wav = torch.istft(SP,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,length=hop_length*(S.size(2)-1)+n_fft)
    return wav

def wav_preprocess(wav,n_fft,hop_length):
    stft = torch.stft(torch.tensor(wav).to(device), n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,return_complex=False)
    stft_mag=torch.abs(torch.sqrt(torch.sum(torch.pow(stft,2),dim=-1)+1e-10))
    stft_phase=(torch.atan2(stft[:,:,1].data, stft[:,:,0].data)/np.pi)
    return stft_mag,stft_phase

def GLA(S,P, n_fft, hop_length, n_iter = 1000):
    P = P * np.pi
    for i in range(n_iter):
        SP = S * torch.complex(torch.cos(P),torch.sin(P))
        wav = torch.istft(SP,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,length=hop_length*(S.size(2)-1)+n_fft)
        next_SP = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,return_complex=False)
        P = torch.atan2(next_SP[:,:,:,1].data, next_SP[:,:,:,0].data)
    return wav

def main():
    args = parser.parse_args()
    Attacker(args)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vox2_data_reader.py
# Copyright (c) 2019 Alvin(Xinyao) Sun <xinyao1@ualberta.ca>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#import matplotlib ##测试用
#matplotlib.use('Qt4Agg') ##测试用
#import matplotlib.pyplot as plt ##测试用

from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
import pandas as pd
#from numpy import random_intel
from pydub import AudioSegment
import imageio
from skimage import io, transform, color
import numpy as np
import os
import glob
import librosa
import librosa.display ##
import matplotlib.pyplot as plt ##

class DataReader(Dataset):
    def __init__(self, csv_meta, audio_prefix, video_prefix, random=False, engine="librosa"):

        print("Init DataReader start")
        self.meta_data = pd.read_csv(csv_meta)
        self.meta_data = self.meta_data[self.meta_data["Set "].str.strip() == "dev"]
        self.ids = self.meta_data["VoxCeleb2 ID "].str.strip().values
        self.ids.sort()
        self.length = len(self.ids)
        self.audio_prefix = audio_prefix
        self.video_prefix = video_prefix
        self.random = random
        self.dur = int((1 / 25) * 16000)
        self.engine = engine

        print("Init DataReader finished")

    def __len__(self):
        # Return total numbe of speakers
        return self.length

    def __getitem__(self, idx):
        ##np.random.seed(int(time.time()*100000)%100000) 暂时固定一下random seed
        np.random.seed(1)
        try:
            speaker_id = self.ids[idx]

            # get all video ids of this speaker
            all_video_pathes, video_ids = self.return_all_vidoes_pathes(speaker_id)
            num_videos = len(all_video_pathes)

            pick_video_idx = 0
            if self.random is True:
                # pick a random video of this speaker
                pick_video_idx = np.random.randint(1, num_videos)

            # get all video clips of selected video
            all_audio_clips = glob.glob(all_video_pathes[pick_video_idx] + "/*.m4a")
            all_audio_clips.sort()
            num_audio_clips = len(all_audio_clips)

            pick_clip_idx = 0
            if self.random is True:
                # pick a random video clip
                pick_clip_idx = np.random.randint(0, num_audio_clips)

            # convert acc to wav
            audio_path_acc = all_audio_clips[pick_clip_idx]
            audio_path_wav = all_audio_clips[pick_clip_idx] + ".wav"
            if os.path.exists(audio_path_wav):
                pass
            else:
                acc = AudioSegment.from_file(audio_path_acc, "m4a")
                acc.export(audio_path_wav, format='wav')

            if self.engine == "librosa":
                data, fs = librosa.load(audio_path_wav, sr=None)

            # locate video
            video_id = video_ids[pick_video_idx]

            video_path = "%s%s/%s/%s.mp4" % (self.video_prefix, speaker_id, video_id, audio_path_wav.split('\\')[-1].split('.')[0])
            vid = imageio.get_reader(video_path,'ffmpeg')
            ##num_frames = vid.get_length()
            num_frames = vid.count_frames()

            # 3s segment
            len_seg = 3

            start_idx = 0

            if self.random is True:
                # pick random segments
                start_idx = np.random.randint(0, num_frames - 3 * 25 - 1)

            frames_seg = [np.expand_dims(color.rgb2gray(vid.get_data(x)), -1) for x in range(start_idx, start_idx + 3 * 25)]

            frames_seg = np.asarray(frames_seg)

            dur = self.dur
            raw_data = data[start_idx * dur:(start_idx + 3 * 25-1 ) * dur]
            ##print("the raw audio data is: {}, length {}".format(raw_data,len(raw_data)))

            if len(raw_data) != 47360:
                print(data.shape)
                print(start_idx)
                print(vid.get_length())
                print(raw_data.shape)
                print(dur)

            #----------------------------------------------------------------
            # 用librosa的STFT transform来获取power spectrogram
            #----------------------------------------------------------------
            if self.engine == "librosa":
                Zxx = librosa.core.stft(raw_data.astype(float), hop_length=10 * 16, n_fft=40 * 16)
                Zxx1 = Zxx**0.3
                D = np.abs(Zxx1)
                ## print 所得的 power spectrogram ##
                librosa.display.specshow(librosa.amplitude_to_db(D,ref=np.max),y_axis='log', x_axis='time')
                plt.title('Power spectrogram')
                plt.colorbar(format='%+2.0f dB')
                plt.tight_layout() 
                
                
            sample = {'frames_seg': frames_seg, 'audio_seg': D, 'raw_data': raw_data,'id':idx}

            return sample

        except Exception as e:
            print(e)
            print("Error")
            #return self.__getitem__(np.random_intel.randint(0, self.__len__()))
            return self.__getitem__(np.random.randint(0, self.__len__()))

    def return_all_vidoes_pathes(self, id):
        """
        return all video pathes for a give human id
        """
        all_video_pathes = glob.glob(self.audio_prefix + id + "/*")
        all_video_pathes.sort()
        video_ids = [i.split("\\")[-1] for i in all_video_pathes]
        return all_video_pathes, video_ids

    
    
    
    
from mag_sub import magnitude_subnet
from phase_sub import phase_subnet

if __name__ == "__main__":
    
    # Change PATHs to you local PATHs
    CSV_META = "./vox2/vox2_meta_small.csv" # "test with modified small scale meta csv"
    VIDEO_PREFIX = "./vox2/vox2_dev_mp4/dev/mp4/" # "change to your own local path"
    AUDIO_PREFIX = "./vox2/vox2_aac/dev/aac/" # "change to your own local path"
    BATCH_SIZE = 4
    db = DataReader(
                csv_meta = CSV_META,
                audio_prefix = AUDIO_PREFIX,
                video_prefix= VIDEO_PREFIX,
                random = True,
                engine = "librosa"
                )
        
    print("==================================")
    print("Test signle output:")
    test_single_input = db[0]


    #original = test_single_input["raw_data"]
    #D = librosa.core.stft(original.astype(float), hop_length=10 * 16, n_fft=40 * 16)
    
    ### 把这个power spectrogram分成magnitude和phase spectrogram两部分
    #magnitude, phase = librosa.core.magphase(D)
    ###print("magnitude spectrogram is {}\nphase spectrogram is {}".format(magnitude,phase))
    
    ## Feed magnitude spectrogram into magnitude subnetwork
    #mag_result = magnitude * magnitude_subnet(magnitude, test_single_input["frames_seg"])
    ###print(">>the result after magnitude subnetwork is: {}".format(mag_result))
    
    ## Feed phase spectrogram into phase subnetwork
    #phase_result = phase + phase_subnet(phase, mag_result)
    ###print(">>the result after magnitude subnetwork is: {}".format(phase_result))
    
    ## Feed modified magnitude and phase spectrogram into ISTFT to get audio signal
    #inverse_result = librosa.core.istft(mag_result*phase_result)
    ###print(">>>>>>>>>>>>>>calculated ISTFT result = {}".format(inverse_result))
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print("Shape of video frames: " , test_single_input["frames_seg"].shape)
    print("Shape of audio spectrogram: ", test_single_input["audio_seg"].shape)
    print("Shape of raw audio wave: ", test_single_input["raw_data"].shape)

    print("==================================")
    print("Test Batch dataloader output: with batch_size %d" % BATCH_SIZE)

    db_loader = DataLoader(db, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(db_loader):
        print("num_batch %d"%i_batch)
        print("shape of frames_seg batch:", sample_batched['frames_seg'].shape)
        print("shape of audio_seg batch:", sample_batched['audio_seg'].shape)
        print("shape of raw_data batch:", sample_batched['raw_data'].shape)
        print("")
        if i_batch == 3:
            break
        
        

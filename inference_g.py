# Copyright (c) 2020, Soohwan Kim. All rights reserved.
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

import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
import os
from torch import Tensor
from tqdm import tqdm

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)


def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'wav') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=20,
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

def revise(sentence):
    words = sentence[0].split()
    result = []
    for word in words:
        tmp = ''    
        for t in word:
            if not tmp:
                tmp += t
            elif tmp[-1]!= t:
                tmp += t
        if tmp == '스로':
            tmp = '스스로'
        result.append(tmp)
    return ' '.join(result)

def addText(lst):
    for elem in lst:
        elem = "chunk"+str(elem)+".wav"
    return lst

parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)
parser.add_argument('--device', type=str, required=False, default='cpu')
opt = parser.parse_args()


vocab = KsponSpeechVocabulary('data/vocab/aihub_labels.csv')
model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(opt.device)
sentences = []

path_test_wav = opt.audio_path
for dir in os.listdir(path_test_wav):
    test_folder = []
    path = os.path.join(path_test_wav,dir)
    for file in os.listdir(path):
        # 파일을 순서대로 읽어오도록 함
        if file.endswith(".wav"):
            tmp = file.replace("chunk","")
            tmp = tmp.replace(".wav","")
            test_folder.append(int(tmp))
    # 파일명의 숫자를 기준으로 정렬
    test_folder.sort()
    for i in range(len(test_folder)):
        test_folder[i] = "chunk"+str(test_folder[i])+".wav"
    print(test_folder)

    # 각 폴더 내부에 txt,wav 파일들의 이름이 겹치므로
    # inference는 폴더 단위로 진행하여 결과 텍스트를 저장

    for i, test_wav in tqdm(enumerate(test_folder)):
        
        feature = parse_audio(os.path.join(path, test_wav), del_silence=True)
        input_length = torch.LongTensor([len(feature)])

        if isinstance(model, nn.DataParallel):
            model = model.module
        model.eval()

        # greedy_search -> recognize로 변경 + 변수 중에서 opt.device 삭제
        if isinstance(model, ListenAttendSpell):
            model.encoder.device = opt.device
            model.decoder.device = opt.device

            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, DeepSpeech2):
            model.device = opt.device
            y_hats = model.recognize(feature.unsqueeze(0), input_length)
        elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
            y_hats = model.recognize(feature.unsqueeze(0), input_length)

        sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
        sentences.append(revise(sentence))
        with open('/content/drive/Shareddrives/KdataB3/Mallang/곧지움/inference2.txt','w',encoding='UTF-8') as f:
            for name in sentences:
                f.write(name+'\n')






        

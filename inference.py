
from jiwer import wer
# from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import json
 
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import Dataset
import numpy as np
from dataloader import MDD_Dataset
import einops
from dataloader import text_to_tensor
from MDD_model import Wav2Vec2_Teacher, Wav2Vec2_Student_woL, Wav2Vec2_Teacher_woL, Wav2Vec2_Student, Wav2Vec2_Student_InterKD
from pyctcdecode import build_ctcdecoder
from jiwer import wer
import ast
import time
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100

gc.collect()

df_dev = pd.read_csv("./test.csv")

model = Wav2Vec2_Student_InterKD.from_pretrained(
    'facebook/wav2vec2-large-xlsr-53', 
)
model.load_state_dict(torch.load("checkpoint/improve_inter_kd_XLSR_w2v2_student.pth"))

model = Wav2Vec2_Teacher_woL.from_pretrained(
    'facebook/wav2vec2-large-xlsr-53'
)

model.load_state_dict(torch.load("checkpoint/wol_XLSR_w2v2_teacher.pth"))

model.freeze_feature_extractor()
model = model.to(device)
PATH = []
CANONICAL = []
TRANSCRIPT = []
PREDICT = []
list_vocab = ['t ', 'n* ', 'y* ', 'uw ', 'er ', 'ah ', 'sh ', 'ng ', 'ey* ', 'd* ', 'jh* ', 'ow ', 'aw ', 'ao* ', 'aa ', 'z* ', 'dh* ', 'aa* ', 'uw* ', 'th ', 'er* ', 'ih ', 't* ', 'zh ', 'g* ', 'k ', 'y ', 'l ', 'uh ', 'eh* ', 'p* ', 'ow* ', 'ch ', 'w ', 'b ', 'l* ', 'v ', 'ao ', 'w* ', 'aw* ', 'ah* ', 'uh* ', 'zh* ', 's ', 'k* ', 'p ', 'iy ', 'r ', 'ae* ', 'eh ', 'b* ', 'f ', 'n ', 'ay ', 'oy ', 'd ', 'g ', 'ey ', 'err ', 'hh* ', 'dh ', 'ae ', 'v* ', 'r* ', 'hh ', 'm ', 'jh ', 'z ', '']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              # kenlm_model_path = '../mdd.binary'
                              )

time_start = time.time()
with torch.no_grad():
  model.eval().to(device)
  worderrorrate = []
  for point in tqdm(range(len(df_dev))):
    acoustic, _ = librosa.load("../WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
    acoustic = feature_extractor(acoustic, sampling_rate = 16000)
    acoustic = torch.tensor(acoustic.input_values, device=device)
    transcript = df_dev['Transcript'][point]
    canonical = df_dev['Canonical'][point]
    canonical = text_to_tensor(canonical)
    canonical = torch.tensor(canonical, dtype=torch.long, device=device)
    # _, _, _, _, _, _, _, _, logits = model(acoustic, canonical.unsqueeze(0))
    _, _, _, _, _, _, _, _, logits = model(acoustic)
    # _, _, _, logits = model(acoustic)
    logits = F.log_softmax(logits.squeeze(0), dim=1)
    x = logits.detach().cpu().numpy()
    hypothesis = decoder_ctc.decode(x).strip()

    PATH.append(df_dev['Path'][point])
    CANONICAL.append(df_dev['Canonical'][point])
    TRANSCRIPT.append(df_dev['Transcript'][point])
    PREDICT.append(hypothesis)
time_end = time.time()

print(time_end-time_start)

train = pd.DataFrame([PATH, CANONICAL, TRANSCRIPT, PREDICT]) #Each list would be added as a row
train = train.transpose() #To Transpose and make each rows as columns
train.columns=['Path','Canonical', 'Transcript', 'Predict'] #Rename the columns
# train.to_csv("checkpoint/improve_inter_kd_XLSR_w2v2_student.csv")
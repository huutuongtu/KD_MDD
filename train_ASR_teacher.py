
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
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
from MDD_model import Wav2Vec2_Teacher, Wav2Vec2_Teacher_woL
from pyctcdecode import build_ctcdecoder
from jiwer import wer
import ast
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100
torch.manual_seed(0)
gc.collect()

def collate_fn(batch):
    
    with torch.no_grad():
        
        sr = 16000
        max_col = [-1] * 4
        target_length = []
        for row in batch:
            if row[0].shape[0] > max_col[0]:
                max_col[0] = row[0].shape[0]
            if len(row[1]) > max_col[1]:
                max_col[1] = len(row[1])
            if len(row[2]) > max_col[2]:
                max_col[2] = len(row[2])

        cols = {'waveform':[], 'linguistic':[], 'transcript':[], 'error':[], 'outputlengths':[]}
        
        for row in batch:
            pad_wav = np.concatenate([row[0], np.zeros(max_col[0] - row[0].shape[0])])
            cols['waveform'].append(pad_wav)
            row[1].extend([68] * (max_col[1] - len(row[1])))
            cols['linguistic'].append(row[1])
            cols['outputlengths'].append(len(row[2]))
            row[2].extend([68] * (max_col[2] - len(row[2])))
            cols['transcript'].append(row[2])
        
        inputs = feature_extractor(cols['waveform'], sampling_rate = 16000)
        input_values = torch.tensor(inputs.input_values, device=device)
        cols['linguistic'] = torch.tensor(cols['linguistic'], dtype=torch.long, device=device)
        cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device)
        cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
    
    return input_values, cols['linguistic'], cols['transcript'], cols['outputlengths']
  
df_train = pd.read_csv('./train_canonical_error.csv')
df_dev = pd.read_csv("./dev.csv")
train_dataset = MDD_Dataset(df_train)

batch_size = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = Wav2Vec2_Teacher_woL.from_pretrained(
    'facebook/wav2vec2-large-xlsr-53',
)
# model.load_state_dict(torch.load("w2v2_XLSR_MDD.pth"))
model.freeze_feature_extractor()
model = model.to(device)

list_vocab = ['t ', 'n* ', 'y* ', 'uw ', 'er ', 'ah ', 'sh ', 'ng ', 'ey* ', 'd* ', 'jh* ', 'ow ', 'aw ', 'ao* ', 'aa ', 'z* ', 'dh* ', 'aa* ', 'uw* ', 'th ', 'er* ', 'ih ', 't* ', 'zh ', 'g* ', 'k ', 'y ', 'l ', 'uh ', 'eh* ', 'p* ', 'ow* ', 'ch ', 'w ', 'b ', 'l* ', 'v ', 'ao ', 'w* ', 'aw* ', 'ah* ', 'uh* ', 'zh* ', 's ', 'k* ', 'p ', 'iy ', 'r ', 'ae* ', 'eh ', 'b* ', 'f ', 'n ', 'ay ', 'oy ', 'd ', 'g ', 'ey ', 'err ', 'hh* ', 'dh ', 'ae ', 'v* ', 'r* ', 'hh ', 'm ', 'jh ', 'z ', '']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
nll_loss = nn.NLLLoss() #should care about ignore index, need to test more
ctc_loss = nn.CTCLoss(blank = 68)
for epoch in range(num_epoch):
  model.train().to(device)
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, linguistic, labels, target_lengths  = data
    output = labels
    transcript = labels
    # _, _, _, _, _, _, _, _, logits= model(acoustic, linguistic)
    _, _, _, _, _, _, _, _, logits= model(acoustic)
    logits = logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    loss_ctc = ctc_loss(logits, labels, input_lengths, target_lengths)
    loss = loss_ctc
    running_loss.append(loss.item())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    # break
  # scheduler.step()
  print(f"Training loss: {sum(running_loss) / len(running_loss)}")
  if epoch>=7:
    with torch.no_grad():
      model.eval().to(device)
      worderrorrate = []
      for point in tqdm(range(len(df_dev))):
        acoustic, _ = librosa.load("../WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
        acoustic = feature_extractor(acoustic, sampling_rate = 16000)
        acoustic = torch.tensor(acoustic.input_values, device=device)
        transcript = df_dev['Transcript'][point]
        canonical = df_dev['Canonical'][point]
        # canonical = text_to_tensor(canonical)
        # canonical = torch.tensor(canonical).to(device)
        # _, _, _, _, _, _, _, _, logits = model(acoustic, canonical.unsqueeze(0))
        _, _, _, _, _, _, _, _, logits = model(acoustic)
        logits = F.log_softmax(logits.squeeze(0), dim=1)
        x = logits.detach().cpu().numpy()
        hypothesis = decoder_ctc.decode(x).strip()
        # print(hypothesis)
        error = wer(transcript, hypothesis)
        worderrorrate.append(error)
      epoch_wer = sum(worderrorrate)/len(worderrorrate)
      if (epoch_wer < min_wer):
        print("save_checkpoint...")
        min_wer = epoch_wer
        torch.save(model.state_dict(), 'checkpoint/wol_XLSR_w2v2_teacher.pth')
      # with open('wer_base.txt', 'a') as wer_file:
      #   wer_file.write(f"Epoch {epoch}: {epoch_wer}\n")
      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))
      
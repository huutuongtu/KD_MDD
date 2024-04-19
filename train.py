
from jiwer import wer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers import Wav2Vec2FeatureExtractor
import torch, json, os, librosa, transformers, gc
import torch.nn as nn


# string representation of list to list using ast.literal_eval()
# import ast
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
from MDD_model import Wav2Vec2_Error
from pyctcdecode import build_ctcdecoder
from jiwer import wer
import ast
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100

gc.collect()

def collate_fn(batch):
    
    with torch.no_grad():
        
        sr = 16000
        max_col = [-1] * 4
        target_length = []
        for row in batch:
            error = ast.literal_eval(row[3])
            # error = json.loads(error)
            if row[0].shape[0] > max_col[0]:
                max_col[0] = row[0].shape[0]
            if len(row[1]) > max_col[1]:
                max_col[1] = len(row[1])
            if len(row[2]) > max_col[2]:
                max_col[2] = len(row[2])
            if len(error) > max_col[3]:
                max_col[3] = len(error)

        cols = {'waveform':[], 'linguistic':[], 'transcript':[], 'error':[], 'outputlengths':[]}
        
        for row in batch:
            pad_wav = np.concatenate([row[0], np.zeros(max_col[0] - row[0].shape[0])])
            cols['waveform'].append(pad_wav)
            row[1].extend([68] * (max_col[1] - len(row[1])))
            cols['linguistic'].append(row[1])
            cols['outputlengths'].append(len(row[2]))
            row[2].extend([68] * (max_col[2] - len(row[2])))
            cols['transcript'].append(row[2])
            error.extend([2] * (max_col[3] - len(error)))
            cols['error'].append(error)
        
        inputs = feature_extractor(cols['waveform'], sampling_rate = 16000)
        input_values = torch.tensor(inputs.input_values, device=device)
        cols['linguistic'] = torch.tensor(cols['linguistic'], dtype=torch.long, device=device)
        cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device)
        cols['error'] = torch.tensor(cols['error'], dtype=torch.long, device=device)
        cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
    
    return input_values, cols['linguistic'], cols['transcript'], cols['error'], cols['outputlengths']
  
df_train = pd.read_csv('./train_canonical_error.csv')
df_dev = pd.read_csv("./dev.csv")
train_dataset = MDD_Dataset(df_train)

batch_size = 2
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

model = Wav2Vec2_Error.from_pretrained(
    'facebook/wav2vec2-base-100h', 
)
# model = torch.load("checkpoint")
model.freeze_feature_extractor()
model = model.to(device)

list_vocab = ['n* ', 'w ', 'v ', 'jh ', 'uh ', 'zh ', 'v* ', 'n ', 't* ', 'aa* ', 'p* ', 'oy ', 'ch ', 's ', 'err ', 'k ', 'uw* ', 'ey ', 'ao ', 'ay ', 'm ', 'eh* ', 'uh* ', 'y* ', 'ah ', 'aa ', 'ey* ', 'z ', 'ae ', 'zh* ', 'ah* ', 'l* ', 'ow ', 'd* ', 'r ', 'iy ', 'th ', 'b ', 'y ', 'ow* ', 'ih ', 'aw ', 'ao* ', 'p ', 'uw ', 'er* ', 'sh ', 'b* ', 'ng ', 'ae* ', 'd ', 'jh* ', 'hh* ', 'k* ', 'hh ', 'f ', 'eh ', 'z* ', 'w* ', 'er ', 'g* ', 'dh* ', 'g ', 'r* ', 'l ', 'aw* ', 'dh ', 't ']
decoder_ctc = build_ctcdecoder(
                              labels = list_vocab,
                              )

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
nll_loss = nn.NLLLoss(ignore_index = 2)
ctc_loss = nn.CTCLoss(blank = 68)
for epoch in range(num_epoch):
  model.train().to(device)
  running_loss = []
  print(f'EPOCH {epoch}:')
  for i, data in tqdm(enumerate(train_loader)):
    acoustic, linguistic, labels, error_gt, target_lengths  = data
    output = labels
    transcript = labels
    logits, error_classifier = model(acoustic, linguistic)
    logits = logits.transpose(0,1)
    input_lengths = torch.full(size=(logits.shape[1],), fill_value=logits.shape[0], dtype=torch.long, device=device)
    logits = F.log_softmax(logits, dim=2)
    error_classifier    = F.log_softmax(error_classifier, dim = 2)

    loss_nll = nll_loss(error_classifier.reshape(-1, 2), error_gt.reshape(-1))
    loss_ctc = ctc_loss(logits, labels, input_lengths, target_lengths)
    loss = 0.7*loss_nll + 0.3*loss_ctc
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
        acoustic, _ = librosa.load("../EN_MDD/WAV/" + df_dev['Path'][point] + ".wav", sr=16000)
        acoustic = feature_extractor(acoustic, sampling_rate = 16000)
        acoustic = torch.tensor(acoustic.input_values, device=device)
        transcript = df_dev['Transcript'][point]
        canonical = df_dev['Canonical'][point]
        canonical = text_to_tensor(canonical)
        canonical = torch.tensor(canonical, dtype=torch.long, device=device)
        logits, _ = model(acoustic, canonical.unsqueeze(0))
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
        torch.save(model, 'checkpoint_v0_73')
      with open('wer_v0_73.txt', 'a') as wer_file:
        wer_file.write(f"Epoch {epoch}: {epoch_wer}\n")
      print("wer checkpoint " + str(epoch) + ": " + str(epoch_wer))
      print("min_wer: " + str(min_wer))
      

from jiwer import wer
from transformers import Wav2Vec2FeatureExtractor
import torch, json, os, librosa, transformers, gc
import torch.nn as nn
import json
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import numpy as np
from dataloader import MDD_Dataset
from dataloader import text_to_tensor
from MDD_model import Wav2Vec2_Teacher, Wav2Vec2_Student, Wav2Vec2_Student_woL, Wav2Vec2_Teacher_woL, Wav2Vec2_Student_InterKD
from pyctcdecode import build_ctcdecoder
from jiwer import wer
from KD_loss import KD_loss_KL_noblank_inputfix, mse_inputfix
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, padding_side='right', do_normalize=True, return_attention_mask=False)
min_wer = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epoch = 100

student = Wav2Vec2_Student_InterKD.from_pretrained(
    'facebook/wav2vec2-large-xlsr-53',
)

# print(teacher)
print(student)
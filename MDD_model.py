import torch.nn as nn
import torch     
from transformers import HubertModel, Wav2Vec2PreTrainedModel, HubertPreTrainedModel, Wav2Vec2Config, HubertConfig, Wav2Vec2Model, Wav2Vec2Tokenizer
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder, Wav2Vec2FeatureProjection, Wav2Vec2EncoderLayerStableLayerNorm
from transformers.modeling_outputs import CausalLMOutput, SequenceClassifierOutput
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import math
from typing import Any, Dict, List, Optional, Union
import einops
import pytorch_revgrad
import gc
import deepspeed

_HIDDEN_STATES_START_POSITION = 2
pretrain_processor_wav2vec2 = 'facebook/wav2vec2-base-100h'
pretrain_audio_model_wav2vec2 = 'facebook/wav2vec2-base'
pretrain_audio_model_wav2vec2 = 'facebook/wav2vec2-large-xlsr-53'


class LinguisticEncoder(nn.Module):
    def __init__(self, num_features_out=1024, vocab_size=68):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size+1, 64, padding_idx=vocab_size)
        self.bi_lstm    = nn.LSTM(
            input_size=64, hidden_size=num_features_out//2, bidirectional=True, 
            batch_first=True, num_layers=4
        )
        self.linear     = nn.Linear(num_features_out, num_features_out)

    def forward(self, x):
        # x shape : batch_size x length_phoneme, output shape: batch x length x n_features
        x           = self.embedding(x)     # batch_size x length_phoneme x 64
        out, (h_n, c_n)   = self.bi_lstm(x)
        Hk          = self.linear(out)
        Hv          = out
        return Hk, Hv


class Wav2Vec2_Teacher(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.post_init()
        self.classifier_vocab = nn.Linear(2048, 69)
        self.linguistic_encoder = LinguisticEncoder()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, dropout=0.1, batch_first=True)
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input, canonical):
        out = self.wav2vec2(audio_input, 
                            attention_mask=None, 
                            output_hidden_states=True).hidden_states
        
        Hk, Hv = self.linguistic_encoder(canonical)
        o, _ = self.multihead_attention(out[-1], Hk, Hv)

        o = torch.concat([out[-1], o], dim=2)

        logits = self.classifier_vocab(o)
        return out[2], out[5], out[8], out[11], out[14], out[17], out[20], out[-1], logits


class Wav2Vec2_Student(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_hidden_layers = 8
        self.wav2vec2 = Wav2Vec2Model(config)
        self.post_init()
        self.classifier_vocab = nn.Linear(2048, 69)
        self.linguistic_encoder = LinguisticEncoder()
        self.multihead_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, dropout=0.1, batch_first=True)
    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def initialize_weights(self):
        for name, param in self.wav2vec2.encoder.layers.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, audio_input, canonical):
        out = self.wav2vec2(audio_input, 
                            attention_mask=None, 
                            output_hidden_states=True).hidden_states
        
        Hk, Hv = self.linguistic_encoder(canonical)
        o, _ = self.multihead_attention(out[-1], Hk, Hv)

        o = torch.concat([out[-1], o], dim=2)

        logits = self.classifier_vocab(o)
        return out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], logits
    
class Wav2Vec2_Student_InterKD(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_hidden_layers = 8
        self.wav2vec2 = Wav2Vec2Model(config)
        self.post_init()
        self.classifier_vocab = nn.Linear(1024, 69)
        self.classifier_vocab5 = nn.Linear(1024, 69)
        self.classifier_vocab6 = nn.Linear(1024, 69)
        self.classifier_vocab7 = nn.Linear(1024, 69)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def initialize_weights(self):
        for name, param in self.wav2vec2.encoder.layers.named_parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, audio_input):
        out = self.wav2vec2(audio_input, 
                            attention_mask=None, 
                            output_hidden_states=True,
                            return_dict=True)
        
   
        o5 = self.classifier_vocab5(out[5])
        o6 = self.classifier_vocab6(out[6])
        o7 = self.classifier_vocab7(out[7])
        logits = self.classifier_vocab(out[-1])

        return o5, o6, o7, logits
 
class Wav2Vec2_Teacher_woL(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.post_init()
        self.classifier_vocab = nn.Linear(1024, 69)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input):
        out = self.wav2vec2(audio_input, 
                            attention_mask=None, 
                            output_hidden_states=True, 
                            return_dict = True)
        # logits = self.classifier_vocab(out[-1])
        return out.last_hidden_state, out.extract_features
        return out[-1], out[0]
        return out[3], out[6], out[9], out[12], out[15], out[18], out[21], out[24], logits

class Wav2Vec2_Student_woL(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.num_hidden_layers = 8
        self.wav2vec2 = Wav2Vec2Model(config)
        self.post_init()
        self.classifier_vocab = nn.Linear(1024, 69)

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def forward(self, audio_input):
        out = self.wav2vec2(audio_input, 
                            attention_mask=None, 
                            output_hidden_states=True).hidden_states

        logits = self.classifier_vocab(out[-1])
        return out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], logits


# import librosa

# model_name = "facebook/wav2vec2-base-960h"
# tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# model_student = Wav2Vec2_Student_8.from_pretrained(
#     'facebook/wav2vec2-large-xlsr-53',
# )

# model_teacher = Wav2Vec2Model.from_pretrained(
#     'facebook/wav2vec2-large-xlsr-53',
# )

# model_teacher.eval
# model_student.eval()

# #test
# audio_file_path = "sleepiness_141-168_0142.wav"
# y, sr = librosa.load(audio_file_path, sr=16000)
# y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
# audio_input = librosa.to_mono(y_16k)
# inputs = tokenizer(audio_input, return_tensors="pt", padding=True).input_values
# with torch.no_grad():
#     outputs = model_teacher(inputs, output_hidden_states=False)
#     x = model_student.forward_a(inputs)

# """
# """
# import librosa

# model_name = "facebook/wav2vec2-base-960h"
# tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)

# model_teacher = Wav2Vec2_Teacher_woL.from_pretrained(
#     'facebook/wav2vec2-large-xlsr-53',
# )

# model_teacher.eval

# audio_file_path = "sleepiness_141-168_0142.wav"
# y, sr = librosa.load(audio_file_path, sr=16000)
# y_16k = librosa.resample(y=y, orig_sr=sr, target_sr=16000)
# audio_input = librosa.to_mono(y_16k)
# inputs = tokenizer(audio_input, return_tensors="pt", padding=True).input_values
# with torch.no_grad():
#     print(model_teacher(inputs))

# """
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

MSE = F.mse_loss


def blank_frame_elimination(logits: torch.tensor, blank_index: int = 68):
    with torch.no_grad():
        res = torch.argmax(logits, dim=2)
        mask = res != blank_index
    return res, mask.type(torch.LongTensor)


def KD_loss_KL_noblank(src: torch.float32, tgt: torch.float32, temperature: torch.float32) -> torch.float32:
    X = F.log_softmax(src/temperature, dim=2)
    Y = F.softmax(tgt/temperature, dim=2)

    loss = torch.tensor(0).type(torch.float32).to('cuda')
    b = src.shape[1]
    X = einops.rearrange(X, "T N C -> N T C")
    Y = einops.rearrange(Y, "T N C -> N T C")
    r_y, m_y = blank_frame_elimination(logits=Y)
    for i in range(b):
        indices = torch.nonzero(m_y[i]).squeeze(1).to('cuda')
        result_tensor_X = torch.index_select(X[i], 0, indices)
        result_tensor_Y = torch.index_select(Y[i], 0, indices)
        loss += F.kl_div(result_tensor_X.view(-1, 69), result_tensor_Y.view(-1, 69), reduction='batchmean')
    return loss/b


def KD_loss_KL_noblank_inputfix(
    src: torch.float32, tgt: torch.float32, input_lengths: torch.LongTensor, 
    temperature: torch.float32
) -> torch.float32:
    
    X = F.log_softmax(src/temperature, dim=2)
    Y = F.softmax(tgt/temperature, dim=2)

    loss = torch.tensor(0).type(torch.float32).to('cuda')
    b = src.shape[1]
    X = einops.rearrange(X, "T N C -> N T C")
    Y = einops.rearrange(Y, "T N C -> N T C")
    r_y, m_y = blank_frame_elimination(logits=Y)
    l_x = input_lengths
    for i in range(b):
        indices = torch.nonzero(m_y[i]).squeeze(1).to('cuda')
        indices = indices[indices < l_x[i]]
        result_tensor_X = torch.index_select(X[i], 0, indices)
        result_tensor_Y = torch.index_select(Y[i], 0, indices)
        loss += F.kl_div(result_tensor_X.view(-1, 69), result_tensor_Y.view(-1, 69), reduction='batchmean')
    return loss / b


def KD_loss_L2_noblank_inputfix(
    src: torch.float32, tgt: torch.float32, input_lengths: torch.LongTensor
) -> torch.float32:
    
    X = F.softmax(src, dim=2) #time x batch x class
    Y = F.softmax(tgt, dim=2) # time x batch x class

    loss = torch.tensor(0).type(torch.float32).to('cuda')
    b = src.shape[1]
    X = einops.rearrange(X, "T N C -> N T C")
    Y = einops.rearrange(Y, "T N C -> N T C")
    r_y, m_y = blank_frame_elimination(logits=Y)
    l_x = input_lengths
    for i in range(b):
        indices = torch.nonzero(m_y[i]).squeeze(1).to('cuda')
        indices = indices[indices < l_x[i]]
        result_tensor_X = torch.index_select(X[i], 0, indices)
        result_tensor_Y = torch.index_select(Y[i], 0, indices)
        loss += F.mse_loss(result_tensor_X.view(-1, 69), result_tensor_Y.view(-1, 69), reduction='batchmean')
    return loss / b


def mse_inputfix(
    src: torch.float32, tgt: torch.float32, input_lengths: torch.LongTensor
) -> torch.float32:
     
    loss = torch.tensor(0).type(torch.float32).to('cuda')
    b = src.shape[0]
    X = src
    Y = tgt
    l_x = input_lengths
    for i in range(b):
        result_tensor_X = X[i][:l_x[i]]
        result_tensor_Y = Y[i][:l_x[i]]
        loss += F.mse_loss(result_tensor_X, result_tensor_Y)
    return loss / b


def mse_noblank_inputfix(
    src: torch.float32, tgt: torch.float32, input_lengths: torch.LongTensor
) -> torch.float32:
     
    loss = torch.tensor(0).type(torch.float32).to('cuda')
    Y = F.softmax(tgt, dim=2)
    Y = einops.rearrange(Y, "T N C -> N T C")
    r_y, m_y = blank_frame_elimination(logits=Y)
    b = src.shape[0]
    X = src
    Y = tgt
    l_x = input_lengths
    for i in range(b):
        indices = torch.nonzero(m_y[i]).squeeze(1).to('cuda')
        result_tensor_X = torch.index_select(X[i], 0, indices)
        result_tensor_X = result_tensor_X[:l_x[i]]
        result_tensor_Y = torch.index_select(Y[i], 0, indices)
        result_tensor_Y = result_tensor_Y[:l_x[i]]
        loss += F.mse_loss(result_tensor_X, result_tensor_Y)
    return loss / b




"""
Example
# T N C
T = 128
N = 32
C = 69
low = 10  # Lower bound (inclusive)
high = 68  # Upper bound (exclusive)
size = (N,)  # Shape of the tensor, for example, a 3x4 tensor
output = torch.rand(T, N, C)
target = torch.rand(T, N, C)
print(KD_loss_KL_noblank(output, target, temperature=7))
"""
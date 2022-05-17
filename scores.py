import torch
import tqdm 
import numpy as np

import sys 
sys.path.append('./nas-bench-nlp-release')

from utils import repackage_hidden

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

def compute_te_nas(model, dataloader, criterion, batch_size):
    # TODO Remove batch number dependency check out NTK
    model.eval()
    model.zero_grad()
    matrix = []
    hidden = model.init_hidden(batch_size)
    k = 0
    for data, targets in tqdm.tqdm(dataloader):
        output, hidden = model(data, hidden)
        loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
        loss.backward()

        grad = []
        for p in model.parameters():
            if p.grad is not None:
                grad.append(p.grad.view(-1).detach().cpu())



        k += 1
        matrix.append(torch.cat(grad))
        hidden = repackage_hidden(hidden)
        model.zero_grad()
    
    tensor_matrix = torch.stack(matrix, dim=0) / k
    ntk = torch.einsum('nc,mc->nm', [tensor_matrix, tensor_matrix])

    eigenvalues, _ = torch.symeig(ntk)  # ascending
    conds = np.nan_to_num((eigenvalues[-1] / eigenvalues[0]).item(), copy=True, nan=100000.0)
    return conds


def compute_ze_nas(gpu, model, batch_size, repeat=1, mixup_gamma=0.1, batch_len=50, fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            reset_model(model)
            input = torch.randn(size=[batch_size, batch_len, model.encoder.weight.shape[-1]], device=device, dtype=dtype)
            input2 = torch.randn(size=[batch_size, batch_len, model.encoder.weight.shape[-1]], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2
            hidden = model.init_hidden(batch_size)
            output, _ = model(input=None, hidden=hidden, raw_input=input) # removing last layers
            hidden = model.init_hidden(batch_size)
            mixup_output, _ = model(input=None, hidden=hidden, raw_input=mixup_input)

            nas_score = torch.sum(torch.abs(output - mixup_output), dim=-1)
            nas_score = torch.mean(nas_score)

            nas_score = torch.log(nas_score)
            nas_score_list.append(float(nas_score))


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list)


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info


def compute_norm_score(gpu, model, criterion, batch_size, batch_len=50):

    model.requires_grad_(True)
    model.zero_grad()
    model.eval()


    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    reset_model(model)
    input = torch.randn(size=[batch_size, batch_len, model.encoder.weight.shape[-1]], device=device)
    if gpu is not None:
        input = input.cuda(gpu)
    
    hidden = model.init_hidden(batch_size)
    output, _ = model(input=None, hidden=hidden, raw_input=input)

    num_classes = 10000
    targets = torch.randint(low=0, high=num_classes, size=[batch_size], device=device)

    loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)
    loss.backward()
    norm2_sum = 0
    with torch.no_grad():
        for p in model.parameters():
            if hasattr(p, 'grad') and p.grad is not None:
                norm2_sum += torch.norm(p.grad) ** 2

    grad_norm = float(torch.sqrt(norm2_sum))
    model.zero_grad()
    return grad_norm



import torch
from torch import nn

from torch.autograd import grad
import copy

def grad_batch(batch_x:torch.Tensor, 
               num_layers, model, Loss=None,
               reconstruct_num=16, model_name='iTransformer', style='channel'
               ):
    
    grads_encoder=None
    grads_decoder=None
    model = copy.deepcopy(model)
    model.eval()
    loss = 0
    if reconstruct_num > 1:
        model.train()
        for _ in range(reconstruct_num):
            outputs = model(batch_x, None, None, None)
            loss += Loss(outputs, batch_x.float())
        loss /= reconstruct_num
    else:
        model.eval()
        outputs = model(batch_x, None, None, None)
        loss = Loss(outputs, batch_x.float())

    B, L, N = loss.shape
    if style=='channel':
        loss = torch.mean(loss, dim=1).reshape(-1, 1)
    else:
        loss= torch.mean(loss, dim=-1)
        loss = torch.mean(loss, dim=-1)

    if model_name=='iTransformer':
        decoder_params = [p for p in model.projection.parameters() if (p.requires_grad)]
        decoder_params = decoder_params[-num_layers:]

    elif model_name=='PatchTST':
        decoder_params = [p for p in model.head.parameters() if (p.requires_grad)]
 
    grads=[]
    for i in range(loss.shape[0]):
        g = list(grad(loss[i], decoder_params, create_graph=True))
        g = torch.nn.utils.parameters_to_vector(g)
        g = g.detach()
        grads.append([g])
 

    return grads
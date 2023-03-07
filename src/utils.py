from torch import nn
import torch 
from torch.nn import NLLLoss
def collate_fn(batch_list):
    # batch_list : [{query : model_inputs, ctx : model_inputs}, ...]
    # model_inputs : {'input_ids' : torch.tensor, 'attention_mask' : torch.tensor, 'token_type_ids' : torch.tensor}
    # return : {'query' : {'input_ids' : torch.tensor, 'attention_mask' : torch.tensor, 'token_type_ids' : torch.tensor},
    #           'ctx' : {'input_ids' : torch.tensor, 'attention_mask' : torch.tensor, 'token_type_ids' : torch.tensor}}
    batch = {}
    for query_or_ctx in batch_list[0].keys() :
        batch[query_or_ctx] = {}
        for model_input_key in batch_list[0][query_or_ctx].keys() :
            batch[query_or_ctx][model_input_key] = torch.stack([f[query_or_ctx][model_input_key] for f in batch_list]).squeeze()
    return batch
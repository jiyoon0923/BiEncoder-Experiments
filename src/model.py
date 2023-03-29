import logging
from typing import Tuple, List
from collections import deque

from torch import nn
import torch.nn.functional as F
import torch 
from torch.nn import NLLLoss
from transformers import BertConfig, BertModel


logger = logging.getLogger(__name__)

class BiEncoder(nn.Module) :
    """
    Encapsulates Query/Context Encoder and Loss Function
    codes from DPR Official Code(https://github.com/facebookresearch/DPR/blob/main/dpr/models/hf_models.py) 
    modified for additional experience by jaehee_kim@korea.ac.kr
    """
    def __init__(self, 
                query_encoder : nn.Module, 
                ctx_encoder : nn.Module, 
                freeze_query_encoder : bool = False,
                freeze_ctx_encoder : bool = False,
                momentum : float = 0
                ) -> None:
        super().__init__()
        self.loss_fn = NLLLoss()

        self.query_encoder = query_encoder
        self.ctx_encoder = ctx_encoder
        self.freeze_query_encoder = freeze_query_encoder
        self.freeze_ctx_encoder = freeze_ctx_encoder
        self.momentum = momentum
        if self.momentum > 0:
            logger.info(f">>> Context Encoder is Freezed : momentum = {self.momentum}")
            for param in self.ctx_encoder.parameters():
                param.requires_grad = False
    
    @staticmethod
    def get_representation(
            model : nn.Module, 
            input_ids : torch.Tensor, 
            attention_mask : torch.Tensor, 
            token_type_ids : torch.Tensor,
            freeze_model : bool = False, 
        ) -> torch.Tensor:

        if freeze_model :
            with torch.no_grad() :
                outputs = model(
                    input_ids       = input_ids, 
                    attention_mask  = attention_mask, 
                    token_type_ids  = token_type_ids)
                
                pooled_output = outputs['pooled_output']
            
            if model.training :
                pooled_output.requires_grad = True

        else : 
            outputs = model(
                input_ids       = input_ids, 
                attention_mask  = attention_mask, 
                token_type_ids  = token_type_ids)
            
            pooled_output = outputs['pooled_output']
        
        return pooled_output

    def forward(
            self, 
            query_inputs : dict, 
            ctx_inputs : dict, 
            ) :
        query_pooled_repr = self.get_representation(
            model       = self.query_encoder, 
            input_ids   = query_inputs['input_ids'],
            attention_mask = query_inputs['attention_mask'],
            token_type_ids = query_inputs['token_type_ids'],
            freeze_model = self.freeze_query_encoder
            )
        
        ctx_pooled_repr = self.get_representation(
            model       = self.ctx_encoder, 
            input_ids   = ctx_inputs['input_ids'],
            attention_mask = ctx_inputs['attention_mask'],
            token_type_ids = ctx_inputs['token_type_ids'],
            freeze_model = self.freeze_ctx_encoder
            )
        
        return {'query_repr' : query_pooled_repr, 'ctx_repr' : ctx_pooled_repr}

    def get_state_dict(self):
        return self.state_dict()
    
    @torch.no_grad()
    def _momentum_update_ctx_encoder(self) :
        """
        Codes from MoCo Official Code(https://github.com/facebookresearch/moco/blob/main/moco/builder.py)
        """
        for param_q, param_k in zip(self.query_encoder.parameters(), self.ctx_encoder.parameters()) :
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)


class BERTEncoder(BertModel) :
    """
    codes from DPR Official Code(https://github.com/facebookresearch/DPR/blob/main/dpr/models/hf_models.py)
    todo : Generalize this class to support other models (eg. SpanBERT, T5Encoder, etc.)
    """
    def __init__(self, config, project_dim : int = 0) :
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim > 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, model_name : str, dropout : float = 0.1, **kwargs) -> BertModel:
        logger.info("initializing BERT encoder with name %s", model_name)
        cfg = BertConfig.from_pretrained(model_name if model_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        
        return cls.from_pretrained(model_name, config=cfg, **kwargs)

    def forward(
            self, 
            input_ids : torch.tensor,
            attention_mask : torch.tensor,
            token_type_ids : torch.tensor
        ) -> Tuple[torch.tensor, ...]:
        
        outputs = super().forward(
            input_ids       = input_ids, 
            attention_mask  = attention_mask, 
            token_type_ids  = token_type_ids)
        
        sequence_output = outputs['last_hidden_state'] # We don't use pooled_output, rather we use the last hidden state of the [CLS] token

        pooled_output = sequence_output[:, 0, :] # [CLS] token

        if self.encode_proj is not None :
            pooled_output = self.encode_proj(pooled_output)
        
        return {'sequence_output' : sequence_output, 'pooled_output' : pooled_output}


## TODO : add passage-wise loss
class BiEncoderNllLoss(nn.Module) :
    def __init__(self, cfg, train_mode = True) -> None:
        super().__init__()
        if cfg['similarity_function'] == 'cosine' :
            self.similarity_fn = self.cosine_similarity
        elif cfg['similarity_function'] == 'inner_product':
            self.similarity_fn = self.inner_product
        
        if cfg['prebatch_size'] > 0 and train_mode :
            ## codes from DensePhrases Official Code(https://github.com/princeton-nlp/DensePhrases/blob/main/densephrases/encoder.py)
            self.prebatch = deque(maxlen = cfg['prebatch_size'])
            self.prebatch_warmup = cfg['prebatch_warmup']
            self.prebatch_size = cfg['prebatch_size']
            self.update_prebatch = cfg['update_prebatch']
        else :
            self.prebatch = None
        self.loss_fn = NLLLoss()
    
    def forward(self, query_repr : torch.tensor, ctx_repr : torch.tensor, pos_idx = None, global_step = 0) :
        ## concat with prebatch
        if (self.prebatch is not None) and len(self.prebatch) > 0 and (global_step > self.prebatch_warmup): # if prebatch is setup and the training step is larger than the warmup step
            ctx_repr_with_prebatch = torch.cat([ctx_repr, torch.cat(list(self.prebatch))], dim = 0)
        else :
            ctx_repr_with_prebatch = ctx_repr
            
        scores = self.similarity_fn(query_repr, ctx_repr_with_prebatch) # (bsz, bsz)
        softmax_scores = F.log_softmax(scores, dim = 1)
        if pos_idx is None :
            pos_idx = torch.arange(len(query_repr)).to(scores.device)

        max_score_idx = torch.max(scores, dim = 1).indices
        corr_pred = (max_score_idx == pos_idx).sum().item()
        in_batch_acc = corr_pred / len(query_repr)

        ## add prebatch
        if self.prebatch is not None :
            if self.update_prebatch :
                self.prebatch = deque([ctx_repr_with_prebatch[:self.prebatch_size].clone().detach()], maxlen = self.prebatch_size)
            else:
                self.prebatch.append(ctx_repr.clone().detach())
        loss = self.loss_fn(softmax_scores, pos_idx)
        return {'loss' : loss, 'acc' : in_batch_acc}

    def inner_product(self, query_repr, ctx_repr) :
        return torch.matmul(query_repr, torch.transpose(ctx_repr, 0, 1))
    
    def cosine_similarity(self, query_repr, ctx_repr) :
        query_repr = query_repr / torch.norm(query_repr, dim = 1, keepdim = True)
        ctx_repr =  ctx_repr / torch.norm(ctx_repr, dim = 1, keepdim = True)
        return torch.matmul(query_repr, torch.transpose(ctx_repr, 0, 1))
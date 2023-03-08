from typing import Dict, Union, Any, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
import os
import random 
import shutil
import omegaconf
from tqdm import tqdm
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from model import BiEncoder, BERTEncoder, BiEncoderNllLoss
from data import DPRDataset
from utils import collate_fn


class DPRTrainer() :
    def __init__(self, config, logger) :
        '''
        Overall Codes are following the original DPR implementation.
        IR Evaluation cannot be done in the Trainer class because of the indexing process. 
        Trainer will save the model checkpoint every config.cktp_save_step. 
        Evaluation will be done in generate_dense_embedding.py and evaluate_dense_retriever.py CLI tools.
        This process is following the original DPR implementation.
        todo 
        1. create_optimizer_and_scheduler : grouped parameter optimization, learning rate scheduler

        '''
        self.config = config

        self.set_seed(self.config.seed)
        self.loss_fn = BiEncoderNllLoss(self.config['train'])

        ## train setup
        self.biencoder = self.get_biencoder()
        self.optimizer, self.lr_scheduler = self.create_optimizer_and_scheduler(self.config.train['num_training_steps'])
        self.train_dataloader = self.get_train_dataloader()
        self.eval_dataloader = self.get_eval_dataloader()
        self.loss_scale = self.config.train['prebatch_size'] + 1 if self.config.train['use_loss_scale'] else 1
        self.max_iteration = self.config.train['num_training_steps'] if self.config.train['num_training_steps'] else len(self.train_dataloader)//self.config.data['train_batch_size'] * self.config.train['num_train_epochs']

        ## log configs for wandb
        config_for_wandb = {}
        for key, value in self.config.items() :
            if isinstance(value, omegaconf.dictconfig.DictConfig) :
                config_for_wandb.update(value)
            else :
                config_for_wandb.update({key : value})

        wandb.init(project = self.config.wandb['project_name'], config = config_for_wandb, name = self.config.wandb['run_name'])
        self.logger = logger

        self.accelerator = Accelerator()
        self.biencoder, self.optimizer, self.lr_scheduler, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.biencoder, self.optimizer, self.lr_scheduler, self.train_dataloader, self.eval_dataloader
            )
        self.device = self.accelerator.device
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        param_groups = self.get_model_param_grouping_for_training(self.biencoder, weight_decay = self.config.train['weight_decay'])
        optimizer = torch.optim.AdamW(param_groups, lr = self.config.train['learning_rate'], eps = self.config.train['adam_epsilon'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = self.config.train['num_warmup_steps'], num_training_steps = num_training_steps)
        return optimizer, scheduler
    
    def get_model_param_grouping_for_training(self, model : nn.Module, weight_decay : float = 0.0) -> List[Dict]:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params' : [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay' : weight_decay
            },
            {   
                'params' : [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay' : 0.0
            }
        ]
        return optimizer_grouped_parameters

    def get_train_dataloader(self) -> DataLoader:
        dataset = DPRDataset(self.config.data, split = 'train')
        return DataLoader(dataset, batch_size = self.config.data['train_batch_size'], shuffle = True, num_workers = self.config.data['dataloader_num_workers'], collate_fn = collate_fn)

    def get_eval_dataloader(self) -> DataLoader:
        dataset = DPRDataset(self.config.data, split = 'dev')
        return DataLoader(dataset, batch_size = self.config.data['eval_batch_size'], shuffle = False, num_workers = self.config.data['dataloader_num_workers'], collate_fn = collate_fn)

    def get_biencoder(self) -> BiEncoder :
        self.query_encoder = BERTEncoder.init_encoder(
            model_name      = self.config.model['model_name'], 
            project_dim     = self.config.model['projection_dim']
            )
        if self.config.model["share_encoder"] == False :
            self.ctx_encoder = BERTEncoder.init_encoder(
                model_name      = self.config.model['model_name'], 
                project_dim     = self.config.model['projection_dim']
            )
        else :
            self.ctx_encoder = self.query_encoder

        biencoder = BiEncoder(
            query_encoder           = self.query_encoder,
            ctx_encoder             = self.ctx_encoder,
            freeze_query_encoder    = self.config.model['freeze_title_encoder'],
            freeze_ctx_encoder      = self.config.model['freeze_content_encoder']
            )
        
        return biencoder

    def train(self):
        self.global_step = 0
        self.global_epoch = 0
        self.biencoder.train()   

        with tqdm(total = self.max_iteration, desc = '>>> Training', dynamic_ncols = True) as pbar :
            while True :
                self.global_epoch += 1
                for batch in self.train_dataloader :

                    self.training_step(batch)
                    
                    pbar.update(1)

                    if self.global_step % self.config.logging['ckpt_save_step'] == 0 :
                        self._save_checkpoint()

                    if self.global_step % self.config.train['eval_step'] == 0 :
                        self._eval()

                    if self.global_step >= self.max_iteration :
                        self.logger.info(f">>> Training finished at step {self.global_step}")
                        break
                if self.global_step >= self.max_iteration :
                    break

    def training_step(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.global_step += 1

        loss_and_acc = self.compute_loss(inputs = inputs, global_step = self.global_step)
        self.accelerator.backward(loss_and_acc['loss'])
        if self.config.train['max_grad_clip_norm'] != 0 :
            self.accelerator.clip_grad_norm_(self.biencoder.parameters(), self.config.train['max_grad_clip_norm'])
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()

        self.log_metrics(loss_and_acc, self.global_step, self.global_epoch)

    def _eval(self) :
        self.eval_loss = []
        self.eval_inbatch_acc = []

        self.biencoder.eval()
        with torch.no_grad() :
            for batch in tqdm(self.eval_dataloader, desc = '>>> Evaluating:', dynamic_ncols = True) :
                loss_and_acc = self.compute_loss(inputs = batch)
                self.eval_loss.append(loss_and_acc['loss'].cpu())
                self.eval_inbatch_acc.append(loss_and_acc['acc'])

        self.biencoder.train()

        metrics = {
            'eval_loss' : np.mean(self.eval_loss),
            'eval_acc' : np.mean(self.eval_inbatch_acc)
        }

        self.log_metrics(metrics, self.global_step, self.global_epoch)
    
    def compute_loss(self, inputs : Dict[str, Union[torch.Tensor, Any]], global_step : int = 0) -> Dict[str, torch.Tensor] :
        outputs = self.biencoder( # model returns query/ctx representation. see model.py@line87
            query_inputs = inputs['query'],
            ctx_inputs   = inputs['ctx']
        )

        loss_and_acc = self.loss_fn(
            query_repr      = outputs['query_repr'],
            ctx_repr        = outputs['ctx_repr'],
            global_step     = global_step,
        )

        loss_and_acc['loss'] = loss_and_acc['loss'] / self.loss_scale # if prebatch used, the loss must be scaled, otherwise the loss will be bigger than expected; which will cause the model to diverge
        return loss_and_acc
        
    def log_metrics(self, metrics, step, epoch) :
        lr = self.lr_scheduler.get_last_lr()[0]
        metrics['lr'] = lr
        wandb.log(metrics, step = step)
        if step % self.config.logging['logging_steps'] == 0 :
            metrics['step'] = int(step)
            metrics['epoch'] = int(epoch)
            metrics_to_str = ', '.join([f"{k} : {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f">>> {metrics_to_str}")
    
    def set_seed(self, seed : int) :
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if use multi-GPU 
        # CUDA randomness
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    def _save_checkpoint(self):
        save_dir = os.path.join(self.config.logging['output_dir'], self.config.wandb['run_name'], f"checkpoint-{self.global_step}[{self.global_epoch}]")
        self.accelerator.save_state(save_dir)
        self.logger.info(f">>> checkpoint saved at {save_dir}")

def main() : 
    import omegaconf
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = 'configs/train/default.yaml')
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)

    logging_dir = os.path.join(config.logging['logging_dir'], config.wandb['run_name'])
    if os.path.exists(logging_dir) == False :
        os.makedirs(logging_dir)
    elif config.logging['overwrite'] == True : 
        shutil.rmtree(logging_dir)
        os.makedirs(logging_dir)
    else :
        raise FileExistsError(f"{logging_dir} already exists. please change logging_dir or run_name in config file")
    
    logging.basicConfig(
        filename    = os.path.join(logging_dir, config.wandb['run_name'] + '.log'),
        format      = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt     = '%m/%d/%Y %H:%M:%S',
        level       = logging.INFO
    )
    logger = logging.getLogger(__name__)

    trainer = DPRTrainer(config, logger)
    trainer.train()
    
if __name__ == "__main__" :
    main()
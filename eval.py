from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import HNSWFaissSearch as HIPFS

from transformers import AutoTokenizer
import torch

from tqdm import trange
import pandas as pd
import datetime
import os, logging
import numpy as np
from typing import List, Dict

from src.model import BiEncoder, BERTEncoder

class CustomBiEncoder :
    def __init__(self, model_path=None, model_name = 'bert-base-uncased', **kwargs) :
        query_encoder = BERTEncoder.init_encoder(
            model_name = model_name,
            project_dim = 128,
        )

        ctx_encoder = BERTEncoder.init_encoder(
            model_name = model_name,
            project_dim = 128,
        )

        self.model = BiEncoder(
            query_encoder = query_encoder,
            ctx_encoder = ctx_encoder,
        ).cuda()

        checkpoint = torch.load(model_path, map_location='cuda:0')
        self.model.load_state_dict(checkpoint) # ---> HERE Load your custom model

        assert 'bert' in model_name.lower(), "Currently, only bert-based models are supported"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Write your own encoding query function (Returns: Query embeddings as numpy array)
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size, desc = '>>> Encoding Queries : '):
                encoded = self.tokenizer(queries[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt')
                query_repr = self.model.get_representation(
                        model = self.model.query_encoder,
                        input_ids = encoded['input_ids'].cuda(),
                        attention_mask = encoded['attention_mask'].cuda(),
                        token_type_ids = encoded['token_type_ids'].cuda(),
                    )
                query_embeddings += query_repr.detach().cpu()

        return torch.stack(query_embeddings)

    # Write your own encoding corpus function (Returns: Document embeddings as numpy array)  
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size, desc = '>>> Encoding Corpus : '):
                titles = [row['title'] for row in corpus[start_idx:start_idx+batch_size]]
                texts = [row['text']  for row in corpus[start_idx:start_idx+batch_size]]
                encoded = self.tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt')
                corpus_repr = self.model.get_representation(
                        model = self.model.ctx_encoder,
                        input_ids = encoded['input_ids'].cuda(),
                        attention_mask = encoded['attention_mask'].cuda(),
                        token_type_ids = encoded['token_type_ids'].cuda(),
                    )
                corpus_embeddings += corpus_repr.detach().cpu()

        return torch.stack(corpus_embeddings)


def main() :
    import omegaconf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type = str, default = 'configs/eval/base.yaml')
    args = parser.parse_args()

    config = omegaconf.OmegaConf.load(args.config)

    logging_dir = os.path.join(config.logging['logging_dir'], config.wandb['run_name'])
    logging.basicConfig(
        filename    = os.path.join(logging_dir, config['saved_path'].split("/")[-1] + '_index' + '.log'),
        format      = '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt     = '%m/%d/%Y %H:%M:%S',
        level       = logging.INFO
    )

    model_path = os.path.join(config['saved_path'], 'pytorch_model.bin')
    model = HIPFS(
        CustomBiEncoder(model_path=model_path)
        )

    index_name = f"dpr_{config['dataset']}_{config.wandb['run_name']}_{config['saved_path'].split('/')[-1]}_{config['test_data_mode']}"
    index_dir = os.path.join(config['saved_path'], index_name)
    if not os.path.exists(index_dir) :
        os.makedirs(index_dir)
    result_dir = os.path.join(config['saved_path'], config['dataset'] + "result.csv")

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(config['dataset'])
    data_path = util.download_and_unzip(url, "./../datasets")

    corpus, queries, qrels = GenericDataLoader(data_path).load(split=config['test_data_mode'])
    corpus_ids, query_ids = list(corpus), list(queries)

    
    ## Indexing the corpus
    logging.info("start whole corpus indexing")
    start = datetime.datetime.now()
    model.index(corpus, score_function="dot")
    end = datetime.datetime.now()
    model.save(index_dir, index_name)
    total_time_taken_indexing = (end - start).total_seconds()
    logging.info(f"end whole corpus indexing, time taken: {total_time_taken_indexing} s")

    retriever = EvaluateRetrieval(model, score_function="dot")
    results = retriever.retrieve(corpus, queries, )
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
    logging.info(f"ndcg: {ndcg}, map: {_map}, recall: {recall}, precision: {precision}")

    result_df = pd.DataFrame(dict(**{"Indexing Time" : total_time_taken_indexing}, **ndcg, **_map, **recall, **precision), index=[index_name])
    result_df.to_csv(result_dir)

if __name__ == '__main__' :
    main()
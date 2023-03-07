from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from beir.datasets.data_loader import GenericDataLoader
from beir import util
from tqdm import tqdm

from dataclasses import dataclass

import random


# dataset = "msmarco"

# #### Download NFCorpus dataset and unzip the dataset
# url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
# out_dir = os.path.join("/project/codes/01_Personal_Study/datasets")
# data_path = util.download_and_unzip(url, out_dir)

class DPRDataset(Dataset) :
    def __init__(self, config, split = 'test') :
        self.config = config
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(config['dataset'])
        data_path = util.download_and_unzip(url, "./datasets")
        corpus, self.queries, qrels = GenericDataLoader(data_path).load(split = split)
        corpus_ids, self.query_ids = list(corpus), list(self.queries)
        self.qrels_corpus = {query_id : [corpus[qrel_corpus_id] for qrel_corpus_id in list(qrels[query_id].keys())] for query_id in tqdm(self.query_ids, desc = '>>> Loading Relative Corpus', dynamic_ncols=True)}
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])

        self.__tokenize_all()

    
    def __tokenize_all(self) : 
        self.corpus = {query_id : [self.tokenizer(corpus_text['title'], corpus_text['text'], return_tensors = "pt", padding = 'max_length', truncation = True, max_length=self.config['passage_max_len']) for corpus_text in corpus_texts] for query_id, corpus_texts in tqdm(self.qrels_corpus.items(), desc = '>>> Tokenizing Relative Corpus', dynamic_ncols=True)}
        self.queries = {query_id : self.tokenizer(query_text, return_tensors = "pt", padding = 'max_length', truncation = True, max_length=self.config['query_max_len']) for query_id, query_text in tqdm(self.queries.items(), desc = '>>> Tokenizing Queries', dynamic_ncols=True)}

    def __len__(self) :
        return len(self.queries)

    def __getitem__(self, idx) :
        query_id = self.query_ids[idx]
        all_context = self.corpus[query_id]
        context = random.choice(all_context)
        return {'query' : self.queries[query_id], 'ctx' : context}

def main() :
    @dataclass
    class DataArgs : 
        dataset : str = "trec-covid"
        test_data_mode : str = "test"
        passage_max_len : int = 256
        query_max_len : int = 64
        batch_size : int = 2
        num_workers : int = 0
        tokenizer_path : str = "bert-base-uncased"

    args = DataArgs().__dict__
    dataset = DPRDataset(args)
    dataloader = DataLoader(dataset, batch_size = args['batch_size'], num_workers = args['num_workers'], shuffle = True)
    for batch in dataloader :
        print(batch)
        print(batch.size)
        break

if __name__ == '__main__' :
    main()
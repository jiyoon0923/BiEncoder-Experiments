# BiEncoder-Experiments
Bi-Encoder Training Experiments based on Various Training Techniques(e.g. Pre Batch, Passage-wise Loss, Gradient Caching, ...)

## Todo List
  - [X] Validation Dataset In-Batch Negative Accuracy Logging
  - [ ] Gradient Caching Implementation
  - [ ] Passage-Wise Loss Implementation
  - [X] PreBatch After Model Warmup Implementation
  - [ ] Multi GPU Setting
  - [ ] Loading Scheduler & Model

## Proposal Papers for Each Techniques
  - PreBatch : [DensePhrases](https://arxiv.org/abs/2012.12624)
  - Passage-Wise Loss : [PAIR](https://arxiv.org/abs/2108.06027)
  - Gradient Caching : [Condenser](https://arxiv.org/abs/2104.08253) & [Gradient Cache](https://aclanthology.org/2021.repl4nlp-1.31/)

% Bert Family Tree
% Maury Courtland (PhD\; www.maury.science)
% March 10, 2020
---
author:
- Maury Courtland
description:
- A description of some BERT-successor models
title:
- The BERT family tree
---

## Talk Overview
- BERT review
- RoBERTa
- DistilBERT
- ALBERT


## Devlin et al. 2018: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
### Background and Approach
- Pre-trained LMs improve performance on many NLP tasks (both sentence and token level)
- 2 current approaches:
    1) feature-based: use (frozen) pre-trained models as feature encoders and let downstream models learn the task, similar to embeddings
    2) fine-tuning: fine-tune pre-trained model parameters directly on downstream task
    - same pre-train objective: "unidirectional" LMs, which limits architecture choices

### Strengths
- Defines new training objective: "masked language model" (inspired by Cloze)
    - Allows new architectures (here, deep bidirectional Transformer) given freedom from unidirectionality LM constraint (**BIG DEPARTURE** from past approaches)
    - Randomly masks some input tokens (15%), task is to predict vocab ID of those tokens
    - Ex. from original Taylor, 1953: "Chickens cackle and _____ quack"
- Avoids training and engineering time for task-specific architectures

### Design Philosophy
- Have minimal architectural changes from pre-trained to downstream task (minimal parameters learned from scratch)

![Applying pre-trained BERT to downstream tasks](BERT_pretrain_to_finetune){#id .class width=300 height=200px}

### Methods
- Built out of transformers (from "Attention Is All You Need" (2017)):

![Transformer architecture](transformer){#id .class width=300 height=250px}

But here, N=L (for "layers"), H is hidden size, A is number of attention heads
- 2 model sizes reported:
    - $BERT_{BASE}$ (L=12, H=768, A=12, Total Parameters=110M) (equiv. to previous SOTA size)
    - $BERT_{LARGE}$ (L=24, H=1024, A=16, Total Parameters=340M)

### Inputs
- Use WordPiece embeddings (essentially sub-word features) with 30k vocab, denoted $E_{token}$
    - Add segment embedding ($E_A$ or $E_B$) and position embeddings ($E_{i}$) to token embeddings
- Use "aggregate sequence representation for classification tasks" token [CLS] (final hidden vector $C$)
- Can handle either a single sentence or a pair (e.g. <Q, A>), separated by [SEP] token (final hidden vector $T_{[SEP]}$)

![BERT input representation](BERT_input){#id .class width=300 height=100px}

### Outputs (training objective): Masked LM
- Final hidden vectors for masked tokens at final transformer layer
- Fed into output softmax over vocab (like normal) and minimize cross-entropy loss
- *BUT*, this creates mismatch between pre-training and fine-tuning (directly against design philosophy)
    - Solution: don't always [MASK] tokens
    - To remedy, they use 80% [MASK], 10% random token, 10% unmasked correct token

### Outputs: Next Sentence Prediction
- Masked LM is good for token-level tasks, but downstream fine-tuning tasks have relationships between sentences (e.g. question answering, natural language inference, etc.)
- Therefore we need a way to evaluate and embed sentence relationships
- Create an output hidden vector from a special appended [CLS] token and (binary) classify whether the 2 training sentences are related or not (final accuracy is 97+% on NSP)
    - 50% of the time, sample adjacent sentences from the corpus (i.e. related)
    - 50% of the time, sample random sentences from the corpus (i.e. unrelated)
- Simple fix, but boosts performance across the board (esp. for QNLI: 3.5% and SQuAD:.6%)

### Fine-tuning
- Attention mechanisms make BERT easily extensible and can capture many tasks (rather than learning representations from scratch, just change weighting of subspace features)
- Just plug-n-play: use whichever task-specific inputs and outputs you want and fine-tune **all** the parameters
    - Input: Use A/B sentence pairs (e.g. sentence and paraphrase, hypothesis and premise, etc.) OR just a dummy null B sentence (e.g. text classification, sequence tagging)
    - Output for token-tasks: use token hidden vectors as input for token-level tasks (e.g. sequence tagging, QA)
    - Output for sentence-tasks: Use $C$ for sentence-level tasks (e.g. entailment, sentiment analysis)
- Relatively inexpensive (all paper tasks fine-tuned in a few hours on 1 GPU)

### Results: Grand-slam SOTA
![SOTA across the board with some impressive gains](BERT_sota){#id .class width=400 height=300px}

### Ablation studies takeaways
- NSP fairly important (especially for QA and NLI)
- Deep bidirectionality very important
    - Also cheaper than separate unidirectional models
    - And strictly more powerful than unidirectional models
- More parameters = better (sometimes *much* = 8%, sometimes *a bit* .4%)
- ELMo-ish approach of concatenating parameters as features is worse (though not by much, best linear combo is .3 worse on 96.1 F1) than a fine-tuning approach (which again, is generally cheaper and more portable)

## Liu et al. 2019: RoBERTa: A Robustly Optimized BERT Pretraining Approach
### Background and Approach
- Lots of other models have been beating BERT's SOTA results
- These models differ along many dimensions
- Therefore, the results comparisons are apples to oranges and BERT needs a fair chance

### Tweaks from BERT
- Dynamically change masking pattern each time a sentence is encountered rather than generating a few variations of what is masked for each input.
- Remove NSP task (unnecessary in their findings)
- Train on max-length (or almost) sequences during the whole training, rather than beginning on short sentences and only training on long sentences at the very end
- Trains model much longer with much bigger batches (shown to be very beneficial) and with more data (*always* good)

### Dynamic Masking Results
- Slightly better than static (though not across the board), **BUT** worth noting that it prevents overfitting so with dynamic masking the model probably could have kept training
![Dynamic Masking Results](dynamic_masking_results)

### What to choose for input samples?
- SEGMENT-PAIR+NSP (BERT clone): Segments (possibly many sentences) <= 512 tokens with an NSP loss
- SENTENCE-PAIR+NSP: Same as above, but with a pair of natural sentences. Given much shorter input length, increase batch size.
- FULL-SENTENCES: Pack contiguous full sentences (possibly across doc boundaries with an extra [SEP] token) <= 512 tokens. No NSP loss.
- DOC-SENTENCES: Same as above, but inputs cannot cross document boundaries

### Input Encoding Results
- NSP doesn't seem to make a large difference. Longer inputs are definitely better.
![Choice of input samples and NSP loss](input_choice)

### Results of Batch Size (Goldilocks)
![This Batch Size is *just* right](batch_size_effect)

### Results of Training Corpus and Length
![Bigger and Longer Training is Better](training_schedule)

### Downstream Fine-tuning Results
![Across the Board SOTA on GLUE](roberta_downstream_results)
- SQuAD and RACE SOTA too!

## Sanh et al. 2018: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
### Background and Approach
- BERT is pretty big and slow
- Network sizes are getting ridiculous (looking at you Microsoft's Turing-NLG: **17B** parameters)
    - Are all those parameters really necessary?
- Maybe we can "compress" the performance with knowledge distillation (Hinton)
    - Smaller size would be good for edge computing

### Knowledge Distillation
- Rather than predict on 1-hot classification, predict to match distribution of "teacher" model
    - Calculates distillation loss as: $L_{ce} = \sum_i t_i * log(s_i)$, where $t_i$ and $s_i$ are prob. of teacher and student
    - Gives richer prediction signal than just 1-hot answer (continuous targets $\in (0, 1)$ for **all** labels)
    - Allows a "student" network to mimic the "teacher" model (monkey see, monkey do)
- Can be done just during training of the base MLM task (madlib) or during the downstream task as well (e.g. NER, NLI, etc.)
- Add in Cosine Similarity loss between the final hidden state vectors before the classifier

### Architecture and Approach
- Same general architecture as BERT but
    - Take *every other* attention layer for warm start (reducing network size by $\approx \frac{1}{2}$)
    - Get rid of token-type embeddings for input (i.e. segment "A" or "B")
    - Get rid of pooler layer (linear $tanh()$ square-matrix layer post-encoder pre-classifier)
- Train using improvements of RoBERTa but with original BERT corpus

### Results
- Reduce size (and storage footprint) by **40%** and make inference time **60%** faster
- Retain up to **97%** of performance
![GLUE results (similar results on SQuAD)](distilbert_results)

### Ablation Studies
- Distillation loss and a warm start are **key**
- Cosine similarity loss is fairly important
- MLM is not that important

![](distilbert_ablation_results)

## Lan et al. 2020: ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
### Background and Approach
- Network sizes are getting ridiculous (*still* looking at you Microsoft's Turing-NLG's **17B** parameters) but show good performance
    - Are all those parameters really necessary?
- Maybe we can share parameters across layers and factor some large matrices
    - Smaller size would be good for edge computing

### Tweaks from BERT
1. Factorizing embedding matrices (which is the largest matrix in the network, other than its mirror in the classifier)
2. Parameter sharing across layers (which saves several square matrices each time)
3. NSP swap out for Sentence Order Prediction (bring back the sentence relationship training objective!)

### Embedding matrix factorization
- BERT (and RoBERTa) tie embedding size ($E$) to hidden space dimensionality ($H$, generally 768)
- Conceptually, conflates 2 concepts:
    1. The word embedding which is meant to capture a *context-indepedent* representation
    2. Hidden layer embeddings are meant to capture a *context-dependent* representation
- Practically, given the necessity of the hidden space to capture complex interactions between multiple words, we would want $H >> E$
- Additionally, $|V|$ is really large so increasing $E$ takes many parameters (only sparsely updated during training)

### Embedding matrix factorization results
- So take the $V \times H$ matrix and decompose into $V \times E$ and $E \times H$ matrices
    - Reduces $O(V \times H)$ to $O(V \times E + E \times H)$
    - Can be significant with large H (especially with shared parameters)

![](albert_embedding_size)

### Parameter sharing results
- All-shared is always non-optimal, but also has *way fewer* parameters (and trains faster)
![](albert_parameter_sharing)

### Sentence Order Prediction Results
- SOP is the best objective and seems to implicitly learn NSP (to a degree)
![](albert_sop)

### Overall Results
- Train on BERT corpus with <=512 segment encoding and A/B embeddings with 30k vocabulary
- 18x fewer parameters
- 1.7x faster training time
    - **BUT** slower inference time
![](albert_overall_results)

## Takeaways
### When to use each model
- Concerned about storage space but not runtime and want best results?
    - ALBERT
- Want good old fashioned BERT?
    - RoBERTa
- Concerned about runtime?
    - DistilBERT

### GLUE Leaderboard at time of ALBERT
![](albert_glue_benchmark)

### RACE and SQuAD at time of ALBERT
![](albert_race_squad_benchmark)

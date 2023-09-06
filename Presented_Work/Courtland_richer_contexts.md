% Richer Contexts
% Maury Courtland (PhD\; www.maury.science)
% November 12, 2019
---
author:
- Maury Courtland
description:
- The rise of deeper and pre-trained models in NLP
title:
- Richer contexts: Better NLP with pre-training and multiple layers of abstraction
---

## Talk Overview
- 30k foot background
- ELMo and deep representations
- BERT and pre-trained models


## 30k foot background: The benefits of pre-training and more information
- Pre-training avoids repeated, costly spin-up time (and core-hours)
    - Arguably what allowed CV to take off (AlexNet/ImageNet)
- Fixed models provide a common out-of-the-box starter model to extend (like a base-class module)
    - Helps researchers have a shared common ground of tools
- What could be wrong with *more* information (**a little troll-y**)
    - More signal (as long as it's worth the extra learning time and complexity) can immensely boost performance


## Peters et al. 2018: Deep contextualized word representations (aka ELMo)
### Background and Approach
- Wants to capture both syntax and semantics as previously done
- Wants to capture variance across contexs (i.e. polysemy) *not* previously done
    - Tokens are assigned representations based on entire sentence
    - Uses bidirectional LSTM with language modeling (LM) objective
    - Thus, ELMo (***E***mbeddings from ***L***anguage ***Mo***dels)

### Strengths
- ELMo representations are deep (a function of all biLM layers, not just top layer)
- Allows distributed meaning encoding
    - Higher states capture context-dependent meaning (good for WSD)
    - Lower states capture aspects of syntax (good for POS tagging)
    - With access to all this information, downstream models can select the relevant dimensions of information for the task (i.e. they are transferable)

### Method
- Start with forward and backward LMs
    - Forward model: $p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k | t_1, t_2, ..., t_{k-1})$
    - Backward model: $p(t_1, t_2, ..., t_N) = \prod_{k=1}^N p(t_k | t_{k+1}, t_{k+2}, ... t_N)$
- Jointly maximize their log likelihoods:
    - ![](BiLM_Objective)

### ELMo's Magic
- ...
- Just combine all the representations computed for each token into a single vector!
    ![](Elmo_representations)
- Previous approaches for contextually aware embeddings use a forward LM and take the last layer of the hidden state to be the embedding

### Choosing a Useful Representation
- Simplest case is to simply select the top layer (as in previous work)
- ***Or***, for each task, adjust the weighting function of each layer:
    ![](Elmo_layer_weighting_function)

    - where $s^{task}$ are softmax-normalized weights and $\gamma^{task}$ allows the downstream model to scale the ELMo vector.
    - sometimes it helps to apply layer normalization before weighting

### Applying the Representations Downstream
- Use the pre-trained biLM and a specified architecture for target task
    1) Run pre-trained (frozen) biLM and store representations for each word
    2) Let target task model learn a linear combination of these representations
- Can include (by concatenation) ELMo representations both in the input (concatenated with a context-independent embedding) and output (concatenated with the top hidden layer)
- Authors find it helpful to add "a moderate amount" of dropout
- Also some cases regularization helped (adding $\lambda||w||_2^2$ to the loss making it close to an average of all layers)

### Results: SOTA!
![ELMo's Improvements Across the board](Elmo_results){#id .class width=300 height=200px}

### Analysis
- ELMo improves over just last layer and is better when linear weights are allowed to vary $\lambda=.001$ (better than $\lambda=1$)
- Isolation of layers as features for WSD and POS tagging tasks reveal different layers capture different information
- ELMo also converges (at higher performance) much quicker than previous approaches

![ELMo is very useful for small training corpora](Elmo_corpus_size){#id .class width=300 height=150px}



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

% Word Embeddings
% Maury Courtland (PhD\; www.maury.science)
% September 10, 2019
---
author:
- Maury Courtland
description:
- An introduction to methods of word embedding for NLP
title:
- Word Embeddings et al.
---

## 30k foot background: Quantifying Words for Computational Use
- Graph-based Relationships (e.g. WordNet and SynSets)
    - Very informative and rich structure
    - Can miss non-obvious relationships
    - Incredibly expensive to collect
- Corpora Statistics
    - Very cheap to collect (nowadays)
    - Prone to sampling error
    - Not straightforward how to extract meaning
- Human Judgements (e.g. MRC)
    - Very rich and abstract judgements
    - Subjective (thus susceptible to noise)
    - Decently expensive to collect

- **Choose the cheap one, grab tons of data to deal with sampling error, and figure out how to extract meaning**

## Talk Overview
- Word2Vec and Friends
- GloVe and global techniques
- ELMo and deep representations

## Mikolov et al. 2013a: Efficient Estimation of Word Representations in Vector Space
### Background
- Motivation
    - Most previous approaches treat words as atomic units
    - Misses related nature of words (all are categorically different)
- Previous Approach Strengths
    - More data (but simple) > More complexity (on less data)
    - Simple, straightforward, and robust
- Previous Shortcomings
    - Marginal data returns are decreasing
    - Data labeling is very expensive

### Paper Goals
- Learn high quality word vectors
    - Learn many types (1m+ vocab)
    - From many tokens (1B+ corpus)
- High dimensional embedding space
    - Words can have "multiple degree of similarity"
- Preserve linear structure between words
    - Enables simple algebra $\overrightarrow{King} - \overrightarrow{Man} + \overrightarrow{Woman} \approx \overrightarrow{Queen}$
    - Neural nets do this better than LSA (latent semantic analysis)
- Be relatively quick training
    - Most previous work took forever to train
    - Neural nets train much quicker than LDA (latent dirichlet allocation)

### Feedforward Neural Net Language Model (Previous work)
- Input layer:
    - 1-hot encodings (with vocab size V) of N previous words concatenated
- Projection layer:
    - Project input using a shared projection matrix into an $N \times D$ layer (D = 50-200 x N)
- Hidden layer:
    - Densely connected from projection layer of size H (H = 50-100 x N)
- Output layer:
    - Softmax (i.e. densely connected) over all vocab words (i.e. of size V)

### Improvements of Current Work
- Use hierarchical (Huffman binary tree) for softmax vocab classification
    - Has advantage that frequent words have fewer parameters to learn (given shorter binary codes)
    - Speeds up both training and testing (avoids calculating and normalizing inner products over the **entire** vocabulary)
<!-- CBOW (continuous because the representation is continuous)-->
- CBOW (Continuous Bag of Words)
    - Projection layer encodes all words (i.e. size = D) in addition to sharing projection matrix
        - Means the word vectors are averaged
        - Loses positional information within the window (hence bag of words)
    - Input layer uses symmetric future/past window
 <!-- Skip-gram -->
 - Skip-gram
    - Uses current word as input to classify (using log-linear classifier) surrounding words
    - Uses same continuous projection layer as CBOW
    - Sample closer words more often (choose a random number from 1 to max_context and predict that many words on each side of the input word)


### Graphical Representations of the Models
![CBOW and Skip-gram Architectures](CBOW_And_Skip_Gram_Architecture.png){#id .class width=300 height=200px}

### Testing
- Uses analogy as the testing metric (i.e. "small":"smaller" as "big":?) using cosine-similarity nearest neighbor
- Testing is all or nothing (i.e. closest vector in embedding space was either the correct answer or not)
- Well-trained models capture notion of capital ("France":"Paris" as "Germany":"Berlin")

### Test Set Dimensions
![Example Relationships in the Test Set](Test_Example_Relations.png){#id .class width=300 height=200px}

### Testing Results
![Results of Various Approaches](Testing_Results.png){#id .class width=300 height=200px}

### Dimensionality vs. Tokens vs. Time Tradeoff
![Performance of Various Training Corpora](Vector_Dims_V_Corpus_Size.png){#id .class width=300 height=200px}

### Dimensions Captured
![Analogies in the Test Set](Learned_Relationships.png){#id .class width=300 height=200px}

- Scores about 60% using their criteria
- Can improve by 10% (absolute) by averaging relationship vectors


## Mikolov et al. 2013b: Distributed Representations of Words and Phrases and their Compositionality
### Motivation
- Improved accuracy and training speedup using subsampling
- Better and faster Noise Contrastive Estimation (NCE) compared with hierarchical softmax
- Non-compositionality of phrases (i.e. $\overrightarrow{Boston~Globe} \neq \overrightarrow{Boston} + \overrightarrow{Globe}$)
    - Test on these new phrasal relations:
        -“Montreal”:“Montreal Canadiens”::“Toronto”:“Toronto Maple Leafs”
- Explore meaningful simple vector additions
    - $\overrightarrow{Russia} + \overrightarrow{river} = \overrightarrow{Volga~River}$
    - $\overrightarrow{Germany} + \overrightarrow{capital} = \overrightarrow{Berlin}$

### Negative Sampling
- Task is to differentiate real co-occurrence pairs from fake ones (GAN-esque)
- Approximately maximizes the softmax (easier to compute than Hier. Softmax)
    - Can be simplified further given task demands
- ![](NCE_Objective)
    - Replaces $log~P(w_O|w_I)$ in the Skip-gram objective
    - For each word $w_I$:
    - Distinguish target word, $w_O$ from fake samples from the noise distribution $P_n(w)$ using logistic regression for k negative samples for each 1 real sample
    - Experimental results show k=5--20 are useful for small datasets, 2--5 for large
- Read Goldberg & Levy 2014 for great derivation of the method

### Subsampling of Frequent Words
- Overly common words (e.g. "the", "a", etc.) are too common, providing less information value (as we'll see in GloVe)
- They therefore become saturated easily (very soon into training) due to exposure frequency
- So subsample to counterbalance frequency effects:

    ![](Subsampling)
    - where $f(w_i)$ is word frequency and $t$ is a threshold (around $10^-5$)

### Results
![Better than the original Word2Vec (faster too)](NEG_Results)


## Pennington et al. 2014: GloVe: Global Vectors for Word Representation
### Background and Drawbacks
- Two main approaches to learn embeddings
1) Global matrix factorization techniques (e.g. LSA)
    - Efficiently makes use of global statistical information
    - Poor at capturing syntactic/semantic relationships = bad vector space geometry
2) Local context window techniques (e.g. Word2Vec)
    - Good at capturing similarity = good vector space geometry
    - Idiosyncratic and underutitilze corpus statistics due to separate context training

### Matrix Factorization Approaches
- Use low-rank approximations to decompose large statistics matrices
- Statistics can vary based on application domain
    - LSA is term-document matrix (words X document_count)
    - HAL (Hyperspace Analogue to Language) is term-term cooccurrence
- Main drawback is frequent words (e.g. "the", "and") have disproportionate effect unrelated to meaningful semantic relationships
- Several transformations had been introduced to address this issue and reduce the dynamic range while preserving information

### GloVe: ***Glo***bal ***Ve***ctors
- Conditional probability matrix $P$ (assymetric):
    - Derived from the co-occurrence matrix $X$ (symmetric)
    - $P_{ij} = P(j|i) = X_{ij}/X_i$
    - i.e. the number of times i and j cooccur over the number of times i cooccurs with anything
- Addresses the disproportional effect of frequent words (e.g. "the") by using conditional probabilities
- Allows relationship to other words to dictate meaning which is intepretable using probe words...

### Probability Relationships of Terms
![Probability Relationships of "Ice" and "Steam"](Glove_Coocurrence_Ratios.png){#id .class width=300 height=200px}

- Allows differentiation along meaningful dimensions:
    - Both mutually related ("water") and unrelated ("fashion") terms have similar ratios

### Developing the Learning Objective
- Want to capture the information in $P_{ik}/P_{jk}$
    - It is a function of 3 terms: i, j, and k

        $F(w_i, w_j, \tilde{w}_k) = P_{ik}/P_{jk}$
    - Where $w \in \mathbb{R}^d$ are word vectors
    and $\tilde{w} \in \mathbb{R}^d$ is a separate context vector
- F could be **so** many things, so we impose constraints:
    1) Linear structure on the vector space:

        $F(w_i - w_j, \tilde{w}_k) = P_{ik}/P_{jk}$
    2) Simplify the mapping from vectors (arguments) to scalar (output):

        $F((w_i - w_j)^T \tilde{w}_k) = P_{ik}/P_{jk}$
        - also isolates vector space dimensions

    3) Symmetry between context and word vectors (an arbitrary distinction):

        $w_i^T\tilde{w}_k + b_i + \tilde{b}_k = log(X_{ik})$

### We're almost there...
- But this model weights all co-occurrences equally (and is undefined when $X_{ik} = 0$)
    - Just zero entries are 75--95% of $X$
    - Rare co-occurrences are highly susceptible to noise
        - Amplified given the use of ratios
- So they cast the objective function as a least squares problem
        ![](Glove_LMS){#id .class width=300 height=50px}

    - where the weighting function $f(X_{ij})$ is:
        ![](Glove_LMS_func){#id .class width=300 height=50px}

### Weighting Function
- Satisfies:
    1) $f(0) = 0$, taking care of the ill-defined nature
    2) $f(x)$ is non-decreasing, discounting rare occurrences
    3) $f(x)$ is not huge for large $x$s, deflating common occurrences
- and looks like:
    ![](Glove_Weighting_Function)

### Results for word analogy: SOTA(ish)
![GloVe Analogy Results](Glove_Analogy_Results){#id .class width=300 height=200px}


### Results for named entity recognition: SOTA(ish)
![Named Entity Recognition on Various Tasks](Glove_NER_Results){#id .class width=300 height=200px}


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
- Also some cases regularization helped (adding $\lambda||w||_2^2$ to the loss)

### Results: SOTA!
![ELMo's Improvements Across the board](Elmo_results){#id .class width=300 height=200px}

### Analysis
![ELMo is very useful for small training corpora](Elmo_corpus_size){#id .class width=300 height=200px}

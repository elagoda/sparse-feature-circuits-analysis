---
title: "Sparce Feature Circuits"
author: "Evgeniya Lagoda"
header-includes:
   - \usepackage{bbm}
    
---

# sparse-feature-circuits-analysis

This project is done as a part of AI Safety Fundamentals Alignment Course at BlueDot. The larger goal of this project is to analyse the result of the paper [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) by Samuel Marks et. al. and run experiements using the experiements of this paper as a base. Currently, this project only contains the explanation of ideas and results of the paper. The paper is interesting from AI Safety research perspective because it proposes methods for feature circuit discovery and proposes techniques for removing unintended behaviours in language models.

# Terminology
The following terminology is used in the paper and is defined in this post. In this section we gather the hyperlinks to the definitions.

 - laguage model Pythia-70M
 - feature
 - feature circuit
 - dictinary learning
 - sparce autoencoder
 - 
 - activation patching
 - attribution patching
 - indirrect effect (IE)
 - criterions to evaluate the discovered circuits:
    - interpretability
    - faithfulness
    - completeness


# Background on Language Models and Sparse Autoencoders

A transformer-type language models (LM) has the following structure.

The article uses model Pythia-70B for most it's experiements, which has the following architecture. 

Typically in a language model, a single neuron activates for a variety of contexts. This is know as superposition, and leads to difficulties interpretation of mechanics of the language model. Recently, researchers started exploring usage of dictionary learning techinies to overcome interpretability difficulties. In dictionary learning, one represents original signal in a higher dimensional space, which has more redundancy that the original space.

An autoencoder is a deep learning model whose training goal is to encode given (vector) repsentation of the data into higher or lower space. In the case of dictionary learning in language models, one uses autoencoders with a hidden layer that is several times higher than the number of tokens, $d_model$. During training of a sparse autoencoder (SAE), one uses loss functions that encourage sparseness of representation, that is, representations that have most (preferably all but one) entries zero. Whenever we have a representation that has only one non-zero entry, it is called a feature. Thus a feature can be identified with a scalar valued function, with a scalar value being the value of activation in that particular non-zero entry. In the article, the SAEs have the following parameters
$$W_E\in \mathbb R^{d_{SAE}\times d_{model}}, W_D\in \mathbb R^{d_{model}\times d_{SAE}}, \pmb b_E\in \mathbb R^{d_{SAE}}, \pmb b_D\in \mathbb R^{d_{model}} ,$$
where the columns of $W_D$ are enforced to be unit vectors. Given an input representation $x\in \mathbb R^{d_{model}}, the SAE representations are computed via
$$f(\pmb x)=[f_1(\pmb x), \dots, f_d_{SAE}(\pmb x)]=W_E(\pmb x-\pmb b_D)+ \pmb b_E $$.




Sparce autoencoders are typically trained on the representations obtained after attention block, mlp block, or at a particular index of residual stream. 

Features in different 


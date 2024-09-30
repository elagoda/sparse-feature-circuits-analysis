
# sparse-feature-circuits-analysis

This project is done as a part of AI Safety Fundamentals Alignment Course at BlueDot. The larger goal of this project is to analyse the result of the paper [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) by Samuel Marks et. al. and run additional experiements using the experiements of this paper as a base. Currently, this project only contains the explanation of ideas and results of the paper. The paper is interesting from AI Safety research perspective because it proposes methods for feature circuit discovery and proposes techniques for removing unintended behaviours in language models. The project intends to build on this paper for mechanistic interpretability research.

# Terminology
The following terminology is used in the paper and is defined in this post.

 - [laguage model Pythia-70M](#background-on-language-models-and-sparse-autoencoders)
 - [feature](#background-on-language-models-and-sparse-autoencoders)
 - [feature circuit](#background-on-language-models-and-sparse-autoencoders)
 - [dictionary learning](#background-on-language-models-and-sparse-autoencoders)
 - [sparce autoencoder](#background-on-language-models-and-sparse-autoencoders)
 - [activation patching](#indirect-effect)
 - [attribution patching](#indirect-effect)
 - [indirrect effect (IE)](#indirect-effect)
 - criterions to evaluate the discovered circuits:
    - interpretability
    - faithfulness
    - completeness


# Background on Language Models and Sparse Autoencoders

A transformer-type **language models** (LM) has the following structure.

The article uses model **Pythia-70B** for most it's experiements, which has the following architecture. 

Typically in a language model, a single neuron activates for a variety of contexts. This is known as **superposition**, and leads to difficulties interpretation of mechanics of the language model. Recently, researchers started exploring usage of dictionary learning techinies to overcome interpretability difficulties. In **dictionary learning**, one represents original signal in a higher dimensional space, which has more redundancy that the original space.

An **autoencoder** is a deep learning model whose training goal is to encode given (vector) repsentation of the data into higher or lower space. In the case of dictionary learning in language models, one uses autoencoders with a hidden layer that is several times higher than the number of tokens, $d_model$. During training of a **sparse autoencoder** (SAE), one uses loss functions that encourage sparseness of representation, that is, representations that have most (preferably all but one) entries zero. Whenever we have a representation that has only one non-zero entry, it is called a **feature**. Thus a feature can be identified with a particular neuron of a SAE hidden layer and, at the same time, with a scalar valued function, with a scalar value being the value of activation in that particular non-zero entry. In the article, the SAEs have the following parameters

$W_E \in {\mathbb{R}}^{d_{SAE}\times d_{model}}$, $W_D \in {\mathbb{R}}^{d_{model}\times d_{SAE}}$, $b_{E} \in \mathbb{R}^{d_{SAE}}$, $b_{D} \in {\mathbb{R}}^{d_{model}}$,

where the columns of $W_D$ are enforced to be unit vectors. Given an input representation $\mathbf{x}\in \mathbb R^{d_{model}}$, the SAE representations are computed via

$f(\mathbf{x})=\[f_1(\mathbf{x}), \dots, f_{d_{SAE}}(\mathbf{x})\]=W_E(\mathbf{x}-\mathbf{b}_D)+ \mathbf{b}_E $. From a given SAE representation $f(\mathbf{x})$, the reconstruction is then defined as 

$\hat {\mathbf x} = W_D f(\mathbf{x}) + \mathbf b_D$. The **reconstruction errors** $\epsilon (\mathbf x)$ are defined as $\epsilon (\mathbf x) = \mathbf x- \hat {\mathbf x}$. 

Sparce autoencoders are typically trained on the representations obtained after attention block, mlp block, or at a particular index of residual stream. 

Activations in features in earlier layer of the model impact activations in features in later, forming **computational graph** $G$ of the model. For the purposes of the article, reconstruction errors $\epsilon (\mathbf x)$ are also viewed as nodes of $G$.  For a given context, some features are more impactful for each other, and, in that case, it is said that these features together with connecting them edges of the computational graph $G$, form a **circuit**. What is meant precisely by "impactful" varies depending on the interest of the researchers. To make precise what is meant by circuits by the authors of the given article, we need to define indirect effect of features and edges. 

## Indirect Effect
**Indirrect Effect** (IE) of a node $u$ in a computational graph is a metric that captures impact of change in the amount of activation in $u$ on a fixed performance metric $m$ of a model. Similarly, IE of an edge $e$ between upstream node $u$ and a downstream node $d$ captures the impact of $e$ on the performance metric $m$. More precisely, suppose we are given a pair of inputs $(x_{clean}, x_{patch})$ from the dataset $D$, we define IE for node $u$ as  

$IE(m; u; x_{clean}, x_{[patch} ) = m(x_{clean}| do(u=u_{patch})) - m(x_{clean})$, 

where $u_{clean} \in \mathbb R$ is the value $u$ takes on $x_{clean}$, $u_{patch} \in mathbb R$ is the value $u$ takes on $x_{patch}$, and $m(x_{clean}| do(u=u_{patch}))$ denotes the value of the metric on $x_{clean}$ with the value at $u$ modified to be $u_{patch}$ instead of $u_{clean}$. The type of modification that happens in  

$m(x_{clean}| do(u=u_{patch}))$,

that is, when we change value at a particular node, is called **activation patching**. Since changing value at a particular node impacts activaltion values at all nodes downstream from it, this cannot be computed in effiently. As a way to overcome this inefficiency, one can approximate activation patching with various methods. The authors of the article employ two methods approximation of $IE$ in their experiments. One of them is via what is known in the literature as **attrubution patching**.

The above definition can be also modified for the case when there is only one inpute $x_{clean}$ instead of a pair. In this case, we can compare the clean activation $u_{clean}$
 against setting the value at that particular node to zero (**zero-ablated**), that is, we compute 
 
 $m(x_{clean}| do(u=0)) - m(x_{clean})$.
 
For an edge $e$ between upstream node $u$ and downstream node $d$, the IE is defined as 

$IE(m; e; x_{clean}, x_{patch}) =  m(x_{clean}| do(d=d(x_{clean}|do(u=u_{patch}))) - m(x_{clean})$,

where $do(d=d(x_{clean}|do(u=u_{patch}))$ means setting the value at $d$ to the value of activation one gets by setting $u$ to the value $u_{patch}$ (and the rest of the activations not downstream of $d$ are set to the activations one gets from the forward pass with $x_{clean}$). In the case when $e$ is an edge between two indices of residual stream, one modifies this definition to substract the impact of edges that come from alternative paths between $u$ and $d$ (via attention and MLP layers$. This definition, again, needs to be approximated to be efficient in computations and, similarly to the definition for nodes, it can be modified for use when there is only one input $x_{clean}$ instead of a pair.



# sparse-feature-circuits-analysis

This project is done as a part of AI Safety Fundamentals Alignment Course at BlueDot. The larger goal of this project is to analyse the result of the paper [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) by Samuel Marks et. al. and run additional experiements using the experiements of this paper as a base. Currently, this project only contains the (brief) explanation of ideas and results of the paper. The paper is interesting from AI Safety research perspective because it proposes methods for feature circuit discovery and proposes techniques for removing unintended behaviours in language models. This project intends to build on this paper for mechanistic interpretability research.

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
    - [interpretability](#criterions-of-circuit-performance)
    - [faithfulness](#criterions-of-circuit-performance)
    - [completeness](#criterions-of-circuit-performance)


# Background on Language Models and Sparse Autoencoders

A transformer-type **language models** (LM) has the following structure.(this will be updated in the future)

The article uses model **Pythia-70M** for most it's experiements, which has the following architecture. 
![Pythia-70M](/presentation_assets/pythia-70m.png)

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

that is, when we change value at a particular node, is called **activation patching**. Since changing value at a particular node impacts activaltion values at all nodes downstream from it, this cannot be computed in effiently. As a way to overcome this inefficiency, one can approximate activation patching with various methods. The authors of the article employ two methods approximation of $IE$ in their experiments. One of them is via what is known in the literature as **attribution patching**.

The above definition can be also modified for the case when there is only one inpute $x_{clean}$ instead of a pair. In this case, we can compare the clean activation $u_{clean}$
 against setting the value at that particular node to zero (**zero-ablated**), that is, we compute 
 
 $m(x_{clean}| do(u=0)) - m(x_{clean})$.
 
For an edge $e$ between upstream node $u$ and downstream node $d$, the IE is defined as 

$IE(m; e; x_{clean}, x_{patch}) =  m(x_{clean}| do(d=d(x_{clean}|do(u=u_{patch}))) - m(x_{clean})$,

where $do(d=d(x_{clean}|do(u=u_{patch}))$ means setting the value at $d$ to the value of activation one gets by setting $u$ to the value $u_{patch}$ (and the rest of the activations not downstream of $d$ are set to the activations one gets from the forward pass with $x_{clean}$). In the case when $e$ is an edge between two indices of residual stream, one modifies this definition to substract the impact of edges that come from alternative paths between $u$ and $d$ (via attention and MLP layers$. This definition, again, needs to be approximated to be efficient in computations and, similarly to the definition for nodes, it can be modified for use when there is only one input $x_{clean}$ instead of a pair.


# Sparse Feature Circuits

We the precise definition of IE, we a ready to define what is meant in the article by (sparse) feature circuits. That is, we describe here how they are computed by the authors of the article.

Set some thresholds $T_N$ and $T_E$ for filtering nodes and edges, respectively. The authors of the article typically used $T_N=0.1$ and $T_E=0.01$ in their experiments. Whether one has pairs of datapoints $(x_{clean}, x_{patch})$ or single data points $x_{clean}$ in a dataset $D$, compute IE for each pair or single data points. In case of paired, templatic data, authors suggest to average the IE of each node or edge. In case of non-templatic data, the IE is summed for each node or edge. Filter the resulting mean or summed IE according to the thresholds $T_N$ and $T_E$. The resulting subgraph induced from the computational graph $G$ is what is meant by a circuit $C$.


## Criterions of Circuit Performance

To evaluate the quality of the circuits the authors found via this method, they used the following criterions: interpretability, faithfulness, and completeness. **Interpretability** was evaluated by crowdworkers (volunteers from ARENA slack channel). 

**Fairthfulness** $F(C)$ was evaluated via the following formula

$F(C)=\dfrac{m(C) - m(\emptyset)}{m(G) - m(\emptyset)}$, 

where for a subgraph $H\subseteq G$ of a full computational graph $G$ of the model, the value $m(H)$ is the average of the metric $m$ computed over each data point in $D$ with all activations outside of the nodes of $H$ (and their downstream nodes$ set to their average value on $D$ (**mean ablated**). In the above formula for $F(C)$, the subsgraph $\emptyset$ is the empty subgraph (that is, every node is mean ablated).

**Completeness** $K(C)$ is computed via the following formula

$K(C)=\dfrac{m(G\backslash C) - m(\emptyset)}{m(G) - m(\emptyset)}$, 

where $G\backslash C$ stands for the graph induced on the complement of the nodes of $C$ in $G$.

# Using circuits to remove unintended signal

The authors use [Bias in Bios dataset](https://github.com/Microsoft/biosbias) to train a LM based classifier predicting profession based biographical description. They intentionally trained the model on a biased subset of the dataset where all professors where male and all nurses where female. Then the used the above methodology (the zero-ablation variant) to detect the circuit in the model that is responsible for the classifier accuracy on the training set. By manually inspecting each feature in the circuit, they determined the nodes that are gender relevant. The circuit induced on these nodes formed the final circuit of interest $C$ that is deemed to be reponsible for the bias in the classifier predictions. Then they zero ablated this curcuit and finetuned the model on the training (biased) dataset. After that, they evaluated the model performance on the balanced test set. The model performed comparably to the model trained right away on the balanced data.

# Unsupervised circuit discovery

 The authors propose a method for authomated circuit discovery. This can be done in two steps
 - given a dataset $D$, cluster elements of the dataset based on the SAE representations (concatenate representations for different layers and modules of the LM).
 - for a given cluser, use zero-ablation variant of the technique to discover circuit that is most active on a given cluster.

The authors provide make the autodetected circuits available [online](https://feature-circuits.xyz) : one can inspect the visual representation of a circuit of a cluster and download the contexts of the cluster. The trained sparse autoencoders are also available for examination in [neuropedia](https://www.neuronpedia.org/p70d-sm).

# Future work
Due to time constraints, this presentation is much more consise than I originally intended it to be, so in the future, I will provide more details in this writeup as well as additional (to the examples what the authors provided) examinations of the circuits.

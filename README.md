# CS 224N Final Project - Multitask BERT

This is the starting code for the default final project for the Stanford CS 224N class. You can find the handout [here](https://web.stanford.edu/class/cs224n/project/default-final-project-bert-handout.pdf)

In this project, you will implement some important components of the BERT model to better understanding its architecture. 
You will then use the embeddings produced by your BERT model on three downstream tasks: sentiment classification, paraphrase detection and semantic similarity.

After finishing the BERT implementation, you will have a simple model that simultaneously performs the three tasks.
You will then implement extensions to improve on top of this baseline.

## Setup instructions

* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* There is a detailed description of the code structure in [STRUCTURE.md](./STRUCTURE.md), including a description of which parts you will need to implement.
* You are only allowed to use libraries that are installed by `setup.sh`, external libraries that give you other pre-trained models or embeddings are not allowed (e.g., `transformers`).

## Handout

Please refer to the handout for a through description of the project and its parts.

### Acknowledgement

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

# SMART-minBERT
+ Baseline model. The pretrained minBERT model. The training loss is calculated by the cross-entropy between the predictions on the input and the true labels of the input.
+ Smoothness-inducing adversarial regularized model. Finetuned baseline model with ad- versarial training. We add additional perturbation to the input and introduce this adversarial training loss into the total loss. Tuning parameters in the experiments are: the learning rate in the adversarial training = 1e-3, the iteration variable iter_var ∈ (1, 3), the noise scale of the perturbation ε = 1e-5, and λs ∈ (1, 0.1).
+ Bregman proximal point optimized model. Finetuned baseline model with Bregman proxi- mal point optimization. During training, the model parameters are updated depends on all its predecessors, which are recorded by the Bregman momentum. The tuning hyperparameters are β ∈ (0.1, 0.2) and μ ∈ (0.999, 0.9).

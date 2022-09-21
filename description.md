# Attention Rollout -- RoBERTa

In this demo, we use the RoBERTa language model (optimized for masked language modelling and finetuned for sentiment analysis).
The model predicts for a given sentences whether it expresses a positive, negative or neutral sentiment.
But how does it arrive at its classification?  This is, surprisingly perhaps, very difficult to determine.
A range of so-called "attribution methods" have been developed that attempt to determine the importance of the words in the input for the final prediction;
they provide a very limited form of "explanation" -- and often disagree -- but sometimes provide good initial hypotheses nevertheless that can be further explored with other methods.

Abnar & Zuidema (2020) proposed a method for Transformers called "Attention Rollout", which was further refined by Chefer et al. (2021) into Gradient-weighted Rollout.
Here we compare it to another popular method called Integrated Gradients.

* Gradient-weighted attention rollout, as defined by [Hila Chefer](https://github.com/hila-chefer)
  [(Transformer-MM_explainability)](https://github.com/hila-chefer/Transformer-MM-Explainability/), with rollout recursion upto selected layer
* Layer IG, as implemented in [Captum](https://captum.ai/)(LayerIntegratedGradients), based on gradient w.r.t. selected layer.

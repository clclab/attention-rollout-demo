* Shown on the left are the results from attention rollout, as defined by Abnar & Zuidema (2020)
* In the center are the results from gradient-weighted attention rollout, as defined by [Hila Chefer](https://github.com/hila-chefer)
  [(Transformer-MM_explainability)](https://github.com/hila-chefer/Transformer-MM-Explainability/), with rollout recursion upto selected layer, and split out between contribution towards a predicted positive sentiment and a predicted negative sentiment.
* Layer IG, as implemented in [Captum](https://captum.ai/)(LayerIntegratedGradients), based on gradient w.r.t. selected layer. IG integrates gradients over a path between observed word and a baseline (here we use two popular choices of baseline: the unknown word token, or the padding token).

**Warning**
Both Rollout and IG are so-called "attribution methods". Many such methods have been proposed, all attempting to determine the importance of the words in the input for the final prediction. Note, however, that they only provide a very limited form of "explanation", and that even the best methods often disagree. Attribution methods such as Rollout should not be used as the final word, but as providing initial hypotheses that can be further explored with other methods.

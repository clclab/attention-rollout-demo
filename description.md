# Attention Rollout -- RoBERTa

In this demo, we use the RoBERTa language model (optimized for masked language modelling and finetuned for sentiment analysis).
The model predicts for a given sentences whether it expresses a positive, negative or neutral sentiment.
But how does it arrive at its classification?  This is, surprisingly perhaps, very difficult to determine.

Abnar & Zuidema (2020) proposed a method for Transformers called "Attention Rollout", which was further refined by Chefer et al. (2021) into **Gradient-weighted Rollout**. Here we compare it to another popular method called **Integrated Gradients**.

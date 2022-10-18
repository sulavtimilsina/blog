+++ 
author = "Hugo Authors"
title = "Nepali Language Modeling using RoBERTa"
date = "2022-03-24"
description = "Nepali Bert"
tags = [
    "NLP",
    "Language Model",
    ""]
+++
 
Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. It is at its core a transformer language model with a variable number of encoder layers and self-attention heads.
Here we used RoBERTa model wish is also based on BERT with 12 encoder layers. It builds on BERT and modifies key hyperparameters, removing the next-sentence pretraining objective and training with much larger mini-batches and learning rates.

Since there were not any language models trained on Nepali language, I tried to train it on data collected from Nepali News Portals.
I scrapped about 14.5 GB of text data from more than 50 news sites. We trained two models, one with 128 token lengths and the other with 512 token lengths, using cloud TPUs.  This is an active project, so code will be soon be made open source.

## Find more about this project here. [Link to Project Site](https://nepberta.github.io)

### Other Contributers:
1. [Milan Guatam](https://gautammilan.github.io)
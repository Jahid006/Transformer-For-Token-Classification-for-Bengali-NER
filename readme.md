
# Project Name
Training Pipeline for Transofrmenr Fo Token Classifications. Use the same environment from the base directory.
## Table of Contents
- [FinetuningBERT](#finetuningbert)
- [EvaluatingBERT](#evaluatingbert)
- [FinetuningTinyBERT](#finetuningtinybert)
- [EvaluatingTinyBERT](#evaluatingtinybert)


## FinetuningBERT
1. Change the config file at ./configs/bert_config.py to your need
```python 
import torch
import os


class BertConfig:
    model_name = "bert_ner"  # name of your model
    artifact_path = "./artifact/bert-exp-4"  # where to save trained model, logger output and tensorboard log
    encoder_model = "csebuetnlp/banglabert"  # which huggigface pretrained model to use
    pretrained_model_path = "./artifact/bert-exp-3/epoch_9_vl_0.919374_va_0.858213_rva_0.9079469057059127.pt"  # provide model path if you wish to resume training
    resume_training = True  # flag to resume training
    inf_model_path = "./artifact/bert-exp-3-karim/epoch_7_vl_1.694903_va_0.857500_rva_0.8582013656835682.pt"  # provide model path if you  want to evaluate the model

    authors = ["rifat", "zahidul", "karim"][:1]  # used datasets.
    max_seq_length = 512
    ffn_size = 768
    train_batch_size = 8
    val_batch_size = 8

    out_dropout_rate = 0.2
    start_epoch = 0
    num_train_epochs = 1
    learning_rate = 1e-5

    index_to_class = ["O", "B-PER", "I-PER", "IGNORE"]  # token class list
    class_weights = [0.6, 2, 1.5, 0.5]  # weight if the class
    class_to_index = {c: i for i, c in enumerate(index_to_class)}
    selected_class = ["PER"]
    n_class = len(index_to_class)
    contain_per = {True: "Yes", False: "NO"}
    balance_train_set = True  # to assert uniform trainset in term of PER class

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.abspath(__file__)
```
2. run ```python finetuning_bert.py``` from cmd

## EvaluatingBERT

1. Once again, change the config file at ./configs/bert_config.py to your need
2. run ```python evaluating_bert.py``` from cmd


## FinetuningTinyBERT
1. Change the config file at ./configs/tiny_bert_config.py to your need
```python 
import torch
import os


class TinyBertInfConfig:
    model_name = "tiny_bert_ner"  #model name
    artifact_path = "./artifact/tiny-bert-exp-5"   # where to save the model
    pretrained_model_path = "./pretrained_model/tiny_bert_ner-v3.pt"  # pretrianed tiny_bert model path
    inf_model_path = "./artifact/tiny-bert-exp-1-best/epoch_74_quantized.pt"  # inference model path
    tokenizer_path = "./tokenization/vocab/vocab_bert.4096"  # tokenizer path

    authors = ["rifat", "zahidul", "karim"][:]  # which dataset you wish to used
    vocab_size = 4096

    max_seq_length = 128
    ffn_size = 256
    train_batch_size = 256
    val_batch_size = 256

    out_dropout_rate = 0.2
    num_train_epochs = 100
    learning_rate = 5e-6

    index_to_class = ["O", "B-PER", "I-PER", "IGNORE"]
    index_to_class_dict = {i: v for i, v in enumerate(index_to_class)}

    class_weights = [0.6, 2, 1.5, 0.5]
    class_to_index = {c: i for i, c in enumerate(index_to_class)}
    selected_class = ["PER"]
    n_class = len(index_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = os.path.abspath(__file__)
```
2. Be careful to add the same tokenizer you used for pretraining 
3. run ```python finetuning_tiny_bert.py``` from cmd

## EvaluatingTinyBERT

1. Once again, Change the config file at ./configs/tiny_bert_config.py to your need
2. run ```python evaluating_tiny_bert.py``` from cmd

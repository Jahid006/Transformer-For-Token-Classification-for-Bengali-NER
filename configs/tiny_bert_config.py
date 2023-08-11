import torch
import os


class TinyBertInfConfig:
    model_name = "tiny_bert_ner"
    artifact_path = "./artifact/tiny-bert-exp-5"
    pretrained_model_path = "./pretrained_model/tiny_bert_ner-v3.pt"
    inf_model_path = "./artifact/tiny-bert-exp-1-best/epoch_74_quantized.pt"
    tokenizer_path = "./tokenization/vocab/vocab_bert.4096"

    authors = ["rifat", "zahidul", "karim"][:]
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

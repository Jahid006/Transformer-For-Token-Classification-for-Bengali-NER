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

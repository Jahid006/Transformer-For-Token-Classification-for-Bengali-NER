import torch
import numpy as np
import random, os
import logging
from datasets import load_metric

metric = load_metric("seqeval")


def seed_every_thing(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)


def setup_logger(path):
    os.makedirs(path, exist_ok=True)
    logging.basicConfig(filename=f"{path}/log.log", level=logging.INFO)
    return logging.info


def compute_metrics(predictions, labels, label_list):
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

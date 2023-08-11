from utils import utils
utils.seed_every_thing(seed_val=37)

import json, random
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from configs.tiny_bert_config import TinyBertInfConfig
from modeling.tiny_bert_ner import build_model
from data_pipeline.data_utils import data_processor, DataGenerator
from data_pipeline.compile_datasets import get_data_split
from trainer import Trainer
from tokenization.text_processor import Tokenizer


cfg = TinyBertInfConfig()
logger = utils.setup_logger(cfg.artifact_path)


def get_model_and_tokenizer(cfg):
    model = build_model(cfg)

    tokenizer = Tokenizer(
        sentencepiece_path=cfg.tokenizer_path,
        max_len=cfg.max_seq_length,
        vocab_size=cfg.vocab_size,
    )
    tokenizer.load()
    tokenizer.initialize()

    return model, tokenizer


def get_dataloder(cfg, tokenizer, data, shuffle=False):
    data = DataGenerator(cfg, data, tokenizer=tokenizer, preprocessor=data_processor)
    # data.preload_data()
    # data.shorten_data(0.01)
    loader = DataLoader(data, shuffle=shuffle, batch_size=cfg.train_batch_size)
    return loader


def read_dataset(splits, authors):
    data_splits = []
    for split in splits:
        data_split = get_data_split(split, authors)
        print(split, len(data_split))
        data_splits.append(data_split)

    return data_splits


if __name__ == "__main__":
    train, val, test = read_dataset(["train", "val", "test"], cfg.authors)

    model, tokenizer = get_model_and_tokenizer(cfg)

    train_loader = get_dataloder(cfg, tokenizer, train, shuffle=True)
    validation_loader = get_dataloder(cfg, tokenizer, val, shuffle=False)
    test_loader = get_dataloder(cfg, tokenizer, test, shuffle=False)

    trainer = Trainer(
        config=cfg,
        model=build_model(cfg),
        train_loader=train_loader,
        val_loader=validation_loader,
        printer=logger,
        summary_writter=SummaryWriter(cfg.artifact_path),
        metric=utils.compute_metrics,
    )

    trainer.to_device()
    trainer.configure_optimizer()
    # trainer.from_pretrained(
    #     "./artifact/tiny-bert-exp-4-ner-pretrained/epoch_96_vl_56.742954_va_0.835698_rva_0.7188831045906294.pt"
    # )
    trainer.train(start_epoch=0, epochs=cfg.num_train_epochs)
    (
        validation_accuracy,
        req_validation_accuracy,
        validation_loss,
        predictions,
        ground_truths,
        p_scores
    ) = trainer.run_val_epoch(test_loader)

    print(validation_accuracy, req_validation_accuracy, p_scores)

from utils import utils
utils.seed_every_thing(seed_val=37)

import json, random, shutil
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter

from configs.bert_config import BertConfig
from modeling.bert_ner import build_model
from data_pipeline.data_utils import data_processor, DataGenerator
from data_pipeline.custom_sampler import CustomDatasetSampler
from data_pipeline.compile_datasets import get_data_split
from trainer import Trainer
from transformers import AutoTokenizer


cfg = BertConfig()
logger = utils.setup_logger(cfg.artifact_path)


def get_model_and_tokenizer(cfg):
    model = build_model(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_model)
    return model, tokenizer


def get_dataloder(cfg, tokenizer, data, shuffle=False, sampler=False):
    data = DataGenerator(cfg, data, tokenizer=tokenizer, preprocessor=data_processor)
    # data.preload_data()
    # data.shorten_data(0.01)
    if sampler:
        sampler = CustomDatasetSampler(
            data,
            num_samples=min(len(data), 32000),
            labels=[
                cfg.contain_per[len([tag for tag in d if tag.endswith("PER")]) > 0]
                for d in data
            ],
        )
        loader = DataLoader(data, batch_size=cfg.train_batch_size, sampler=sampler)
    else:
        loader = DataLoader(data, shuffle=shuffle, batch_size=cfg.train_batch_size)
    logger(f"Instance in Dataloader: {len(loader)}")
    return loader


def read_dataset(splits, authors):
    data_splits = []
    for split in splits:
        data_split = get_data_split(split, authors)
        print(split, len(data_split))
        data_splits.append(data_split)

    return data_splits


if __name__ == "__main__":
    shutil.copyfile(cfg.config_path, f"{cfg.artifact_path}/bert_config.py")

    train, val, test = read_dataset(["train", "val", "test"], cfg.authors)
    model, tokenizer = get_model_and_tokenizer(cfg)
    train_loader = get_dataloder(
        cfg, tokenizer, train, shuffle=True, sampler=cfg.balance_train_set
    )
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
    if cfg.resume_training:
        trainer.from_pretrained(cfg.pretrained_model_path)
    trainer.train(start_epoch=cfg.start_epoch, epochs=cfg.num_train_epochs)
    (
        validation_accuracy,
        req_validation_accuracy,
        validation_loss,
        predictions,
        ground_truths,
        p_scores,
    ) = trainer.run_val_epoch(test_loader)

    print(f"{validation_accuracy=}, {req_validation_accuracy=}")
    print("Seqeval Results: ", p_scores)

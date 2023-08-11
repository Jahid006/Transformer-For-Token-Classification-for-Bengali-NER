from itertools import compress
from tqdm import tqdm
import numpy as np
import torch


def data_processor(
    text,
    labels,
    tokenizer,
    class_to_index,
    selected_class,
    max_seq_len,
    device,
    inference_mode=False,
):
    if inference_mode and not labels:
        labels = ["O"] * len(text)  # to do

    assert len(text) == len(labels)
    input_ids, label_ids = [], []

    idx = 0
    token_to_word = [None]

    for word, label in zip(text, labels):
        word_tokens_ids = tokenizer.encode(word, add_special_tokens=False)

        if len(input_ids) + len(word_tokens_ids) - 2 > max_seq_len:
            break

        cls_name = label.split("-")[-1].strip()

        if cls_name in selected_class:
            word_label_ids = [class_to_index[f"B-{cls_name}"]] + [
                class_to_index[f"I-{cls_name}"]
            ] * (len(word_tokens_ids) - 1)
        else:
            word_label_ids = [class_to_index["O"] for i in range(len(word_tokens_ids))]

        label_ids.extend(word_label_ids)
        input_ids.extend(word_tokens_ids)
        token_to_word.extend([idx] * len(word_label_ids))
        idx += 1
    assert len(input_ids) == len(label_ids)

    attention_mask_len = len(input_ids) + 2
    input_ids = (
        [tokenizer.cls_token_id]
        + input_ids
        + [tokenizer.pad_token_id]
        + [tokenizer.pad_token_id] * (max_seq_len - len(input_ids) - 2)
    )
    label_ids = [-100] + label_ids + [-100] * (max_seq_len - len(label_ids) - 1)

    attention_mask = torch.zeros(len(label_ids))
    attention_mask[:attention_mask_len] = 1

    preocessed_data = {
        "input_ids": torch.LongTensor(input_ids).to(device),
        "label_ids": torch.LongTensor(label_ids).to(device),
        "attention_mask": torch.Tensor(attention_mask).to(device),
    }

    if inference_mode:
        for k, v in preocessed_data.items():
            preocessed_data[k] = v.unsqueeze(0)

        preocessed_data["words"] = text
        preocessed_data["word_index"] = token_to_word
    preocessed_data["attention_mask_len"] = attention_mask_len

    return preocessed_data


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        data,
        tokenizer,
        preprocessor=data_processor,
        printer=print,
    ):
        self.dataset = data
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.class_to_index = {c: i for i, c in enumerate(config.index_to_class)}
        self.max_seq_length = config.max_seq_length
        self.selected_class = config.selected_class
        self.device = config.device
        self.printer = printer
        self.preloaded_data = None

        print(f"Total {len(data)} Data found!!!")

    def shorten_data(self, dataset_split=0.1):
        self.printer("Shortening The Dataset to:", str(100 * dataset_split) + "%")

        selected_idx = np.arange(len(self.dataset))
        np.random.shuffle(selected_idx)

        selected_idx = selected_idx[: int(dataset_split * len(selected_idx))]
        self.dataset = list(compress(self.dataset, selected_idx))

    def preload_data(self):
        self.dataset = [
            (
                text,
                label,
                self.preprocessor(
                    text,
                    label,
                    self.tokenizer,
                    self.class_to_index,
                    self.selected_class,
                    self.max_seq_length,
                    self.device,
                ),
            )
            for text, label in tqdm(self.dataset)
        ]
        self.preloaded_data = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.preloaded_data:
            text, label, preocessed_data = self.dataset[index]
        else:
            text, label = self.dataset[index]
            preocessed_data = self.preprocessor(
                text,
                label,
                self.tokenizer,
                self.class_to_index,
                self.selected_class,
                self.max_seq_length,
                self.device,
            )
        try:
            assert len(preocessed_data["input_ids"]) == len(
                preocessed_data["label_ids"]
            ), (
                f"input_ids ({len(preocessed_data['input_ids'])})"
                + f"must be equal to label_ids ({len(preocessed_data['label_ids'])})"
            )
        except Exception as e:
            # self.printer(e, text, label)
            return self[index + 1]

        return preocessed_data

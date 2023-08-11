import sys
import json
import random
from sklearn.model_selection import train_test_split

DATA_DIR = "/mnt/JaHiD/Zahid/RnD/BengaliNamedEntityRecognition/opensource_dataset/datasets"


def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, ensure_ascii=False)

def process_b_ner_data():
    path = f"{DATA_DIR}/B-NER/B-NER.json"
    data = json.load(open(path, 'r'))

    train, test = train_test_split(data, test_size=.3, random_state=37)
    val, test = train_test_split(test,test_size=.5, random_state=37)

    print(len(train), len(test), len(val))

    save_json(path.replace('.json', '_train.json'), train)
    save_json(path.replace('.json', '_test.json'), test)
    save_json(path.replace('.json', '_val.json'), val)


def process_Bengali_NER_data():
    path = f"{DATA_DIR}/Bengali-NER/Bengali-NER.json"
    data = json.load(open(path, 'r'))

    train, val = train_test_split(data['Train'], test_size=.2, random_state=37)

    save_json(path.replace('.json', '_train.json'), train)
    save_json(path.replace('.json', '_test.json'), data['Test'])
    save_json(path.replace('.json', '_val.json'), val)


def process_Bengali_NER_data():
    path = f"{DATA_DIR}/NER-Bangla-Dataset/Bangla-NER-Splitted-Dataset.modified.json"
    data = json.load(open(path, 'r'))

    train, val = train_test_split(data['Train'], test_size=.2, random_state=37)

    save_json(path.replace('.json', '_train.json'), data['Train'])
    save_json(path.replace('.json', '_test.json'), data['Test'])
    save_json(path.replace('.json', '_val.json'), data['Val'])


paths = {
    "karim": f"{DATA_DIR}/__Karim__NER-Bangla-Dataset/Bangla-NER-Splitted-Dataset.modified.json",
    "rifat": f"{DATA_DIR}/__Rifat__Bengali-NER/Bengali-NER.json",
    "zahidul": f"{DATA_DIR}/__Zahidul__B-NER/B-NER.json"
}


def get_data_split(split, authors=None):
    if not authors:
        authors = list(paths.keys())
    data = []
    for p in authors:
        path = paths[p]
        data += json.load(open(path.replace('.json', f'_{split}.json'), 'r'))

    return data

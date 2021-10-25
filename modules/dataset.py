import re
import json

import nltk
import torch
import pandas as pd

from itertools import chain

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast


def preprocess(json_path, mode):

    nltk.download('punkt')

    fd = json.load(open(json_path, encoding='utf-8'))

    if mode == 'train':

        train = pd.DataFrame(columns=['uid', 'title', 'region', 'total', 'summary'])
        uid = 1000
        for data in uid:
            for agenda in data['context'].keys():
                total = []
                contexts = []
                for line in data['context'][agenda]:
                    contexts.append(line)
                for context in contexts:
                    split_context = nltk.sent_tokenize(context)
                    for cnt in split_context:
                        if len(cnt) < 15:
                            continue
                        elif len(cnt) > 650:
                            continue
                        else:
                            total.append(cnt)

                total = list(map(lambda x: re.sub(' {2, }', ' ', x), total))

                train.loc[uid, 'uid'] = uid
                train.loc[uid, 'title'] = data['title']
                train.loc[uid, 'region'] = data['region']
                train.loc[uid, 'total'] = [data['title']] + total
                train.loc[uid, 'summary'] = data['label'][agenda]['summary']
                uid += 1
        return train
    else:
        test = pd.DataFrame(columns=['uid', 'title', 'region', 'total'])
        uid = 2000
        for data in fd:
            for agenda in data['context'].keys():
                total = []
                contexts = []
                for line in data['context'][agenda]:
                    contexts.append(line)
                for context in contexts:
                    split_context = nltk.sent_tokenize(context)
                    for cnt in split_context:
                        if len(cnt) < 15:
                            continue
                        elif len(cnt) > 650:
                            continue
                        else:
                            total.append(cnt)
                total = list(map(lambda x: re.sub(' {2, }', ' ', x), total))

                test.loc[uid, 'uid'] = uid
                test.loc[uid, 'title'] = data['title']
                test.loc[uid, 'region'] = data['region']
                test.loc[uid, 'total'] = [data['title']] + total
                test.loc[uid, 'summary'] = data['label'][agenda]['summary']
                uid += 1
        return test


class CustomDataset(Dataset):

    def __init__(self, data, mode):

        self.data = data
        self.mode = mode
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-summarization')

        if self.mode == 'train':
            self.inputs, self.labels = self._data_loader()
        else:
            self.inputs = self._data_loader()

    def _data_loader(self):

        inputs = pd.DataFrame(columns=['text'])
        labels = pd.DataFrame(columns=['summary'])

        inputs['text'] = self.data['total']
        labels['summary'] = self.data['summary']

        if self.mode == 'train':

            inputs, labels = self._preprocess(inputs, labels)
            return inputs, labels

        else:

            inputs = self._preprocess(inputs, labels)
            return inputs

    def _preprocess(self, inputs, labels):

        inputs['text'] = inputs['text'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode('<s>' + x[i] + '</s>', max_length=int(1024 / len(x)), add_special_tokens=True)for i in range(len(x))]))))
        max_encoding_len = max(inputs.text.map(lambda x: len(x)))
        inputs['text'] = self._pad(inputs.text, self.tokenizer.pad_token_id, max_encoding_len)  # pad token id -> 3
        inputs['mask'] = inputs.src.map(lambda x: ~(x == self.tokenizer.pad_token_id))

        if self.mode == 'train':
            labels['summary'] = labels['summary'].map(lambda x: torch.tensor(self.tokenizer.encode('<s>' + x + '</s>', max_length=150, add_special_tokens=True, truncation=True, padding='max_length')))
            labels['summary_mask'] = labels.summary.map(lambda x: ~(x == self.tokenizer.pad_token_id))
            return inputs.values, labels.values
        else:
            return inputs.values

    def _pad(self, data, pad_id, max_len):

        padded_data = data.map(lambda x: torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, index):

        if self.mode == 'train':
            return [self.inputs[index][i] for i in range(2)], [self.labels[index][i] for i in range(2)]
        else:
            return [self.inputs[index][i] for i in range(2)]

import random

import nltk
import torch
import numpy as np

from transformers import set_seed

nltk.download('punkt')


def fix_seed(seed=5252):

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def accuracy_function(preds, labels, pad_token_id):

    accuracies = torch.eq(preds, labels)
    mask = torch.logical_not(torch.eq(labels, pad_token_id))
    accuracies = torch.logical_and(accuracies, mask)
    accuracies = torch.tensor(accuracies, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.float32)

    return torch.sum(accuracies) / torch.sum(mask)

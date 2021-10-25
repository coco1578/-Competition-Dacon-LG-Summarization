import random

import tqdm
import nltk
import torch
import pandas as pd

# from transformers import

from torch.utils.data import DataLoader
from modules.dataset import CustomDataset, preprocess
from modules.utils import fix_seed

nltk.download('punkt')


def summarize(model, input_ids, tokenizer, max_length=150):

    generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_length=max_length, repetition_penalty=2.0, length_penalty=1.2, early_stopping=True)
    preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds[0]


def predict():

    fix_seed()

    mode = 'test'
    test_df = preprocess('dataset/test.json', mode)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dataset = CustomDataset(test_df, mode)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = torch.load('model.pt')
    model = model.to(device)
    model.eval()

    y_preds = []
    with torch.no_grad():
        for step, (inputs) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
            y_pred = summarize(model, inputs[0].long().to(device), test_dataset.tokenizer)
            print(y_pred)  # for debugging
            y_preds.append(y_pred)
    
    submission = pd.read_csv('dataset/sample_submission.csv')
    submission['summary'] = y_preds
    submission.to_csv('result/baseline.csv', index=False)


if __name__ == '__main__':

    predict()
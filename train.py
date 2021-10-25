import enum
from itertools import accumulate
import torch
import numpy as np

from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import SubsetRandomSampler, DataLoader
from transformers import AutoConfig, BartForConditionalGeneration, AdamW

from modules.model_utils import freeze
from modules.dataset import preprocess, CustomDataset
from modules.utils import fix_seed, postprocess_text, accuracy_function


class CFG:

    seed = 5252
    batch_size = 4
    epoch = 30
    lr = 5e-4
    weight_decay = 0.001
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    freeze_encode = True
    freeze_norm = True


def train():

    fix_seed(CFG.seed)

    mode = 'train'

    train_df = preprocess('dataset/train.json', mode)
    train_dataset = CustomDataset(train_df, mode)

    # split train / validation dataset via summary length
    summary_length = list()
    for i in range(len(train_df)):
        summary_length.append(len(train_df.summary.iloc[i]))

    summary_length = np.array(summary_length)
    long_indices = np.where(summary_length >= 90)[0]
    short_indices = np.where(summary_length < 90)[0]
    long_indices = np.random.permutation(long_indices)
    short_indices = np.random.permutation(short_indices)

    train_short_indicies = short_indices[:-100]
    valid_short_indicies = short_indices[-100:]
    train_long_indicies = long_indices[:-100]
    valid_long_indicies = long_indices[-100:]

    train_idx = np.concatenate((train_short_indicies, train_long_indicies))
    valid_idx = np.concatenate((valid_short_indicies, valid_long_indicies))

    train_subsampler = SubsetRandomSampler(train_idx)
    valid_subsampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, sampler=train_subsampler)
    valid_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, sampler=valid_subsampler)

    config = AutoConfig.from_pretrained('gogamza/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-summarization', config=config)

    if CFG.freeze_encode:
        freeze(model, CFG.freeze_norm)

    optimizer = AdamW(
        params=[
            {'params': [params for name, params in model.model.named_parameters() if 'encoder' in name], 'lr': CFG.lr * 0.01}, 
            {'params': [params for name, params in model.model.named_parameters() if 'decoder' in name and 'encoder' not in name], 'lr': CFG.lr}
        ],
        lr=CFG.lr,
        weight_decay=CFG.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epoch, eta_min=CFG.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.1, div_factor=1e5, max_lr=0.0001, epochs=CFG.epoch, steps_per_epoch=len(train_loader))

    metric = load_metric('rouge')
    save_point = -np.inf

    for epoch in range(CFG.epoch):

        model.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), position=0, leave=True)

        train_loss = 0
        train_acc = list()

        for step, (data, target) in pbar:
            model.train()

            input_ids = data[0].long.to(CFG.device)
            input_attention_mask = data[1].to(CFG.device)
            label_inp = target[0][:, :-1].to(CFG.device)
            label_tar = target[0][:, 1:].to(CFG.device)

            outputs = model(input_ids=input_ids, attention_mask=input_attention_mask, decoder_input_ids=label_inp, labels=label_tar)
            loss = outputs.loss
            logits = outputs.logits

            top1 = logits.argmax(dim=-1)
            acc = accuracy_function(label_tar, top1)

            train_loss += loss
            train_acc.append(acc.item())

            loss = loss / CFG.batch_size
            loss.backward()

            if (step + 1) % CFG.batch_size == 0:
                optimizer.step()
                model.zero_grad()
                scheduler.step()
        
        train_epoch_loss = train_loss / len(train_loader)
        train_epoch_acc = np.mean(train_acc)
        print(f'Epoch {epoch}, Train Loss: {train_epoch_loss}, Train Acc: {train_epoch_acc}')

        model.eval()
        with torch.no_grad():
            pbar = tqdm(enumerate(valid_loader), total=len(valid_loader))

            for step, (data, target) in pbar:
                input_ids = data[0].long.to(CFG.device)
                input_attention_mask = data[1].to(CFG.device)
                label_inp = target[0][:, :-1].to(CFG.device)
                label_tar = target[0][:, 1:].to(CFG.device)

                generated_ids = model.generate(input_ids=input_ids, num_beams=5, max_length=150, repetition_penalty=2.0, length_penalty=1.2, no_repeat_ngram_size=2, early_stopping=True)

                decoded_preds = train_dataset.tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                decoded_labels = train_dataset.tokenizer.batch_decode(label_tar, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

                metric.add_batch(predictions=decoded_preds, references=decoded_labels)

            result = metric.compute(use_stemmer=True)
            result = {key: value.mid.fmeasure for key, value in result.items()}
            result = {k: round(v, 4) for k, v in result.items()}

            r1, r2, rl = result['rouge1'], result['rouge2'], result['rougeL']
            best = np.mean([r1, r2, rl])
            print(f'Epoch {epoch}, Rouge 1: {result["rouge1"]}, Rouge 2: {result["rouge2"]}, Rouge L: {result["rougeL"]}')

            if best > save_point:
                save_point = best
                print(f'Epoch {epoch}, Save Model ...')
                torch.save(model, 'result/model.pt')


if __name__ == '__main__':
    train()

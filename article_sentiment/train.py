#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
import argparse
import logging

import torch
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

from article_sentiment.env import PROJECT_DIR
from article_sentiment.kobert.utils import get_tokenizer
from article_sentiment.kobert.pytorch_kobert import get_pytorch_kobert_model, get_kobert_model
from article_sentiment.data.utils import SegmentedArticlesDataset, BERTDataset
from article_sentiment.data.article_loader import BERTOutputSequence


parser = argparse.ArgumentParser()
parser.add_argument('--device', help="`cpu` vs `gpu`", choices=['cpu', 'gpu'], default='gpu')
parser.add_argument('--fine_tune', help="fine-tune BERT and save output", action='store_true')
parser.add_argument('--fine_tune_save', help="save path for fine-tuned BERT classifier",
                    default=PROJECT_DIR / 'models' / 'bert_fine_tuned.dict', type=str)
parser.add_argument('--fine_tune_load', help="load path for fine-tuned BERT classifier", default='', type=str)
parser.add_argument('--robert', help="train RoBERT", action='store_true')
parser.add_argument('--tobert', help="train ToBERT", action='store_true')
parser.add_argument('--clf_save', help='path to which classifier is saved', default=PROJECT_DIR / 'models' / 'classifier.dict')
parser.add_argument('--test_run', help="test run the code on small sample (2 lines of train and test each)", action='store_true')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('train.py')
logger.setLevel(logging.DEBUG)


class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=4,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler

        return self.classifier(out)


class RoBERT(nn.Module):
    def __init__(self,
                 input_size=768,
                 lstm_hidden_size=100,
                 fc_hidden_size=30,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(RoBERT, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,)
        self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.classifier = nn.Linear(fc_hidden_size, num_classes)
        self.dr_rate = dr_rate

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, seq):
        out, (h_n, c_n) = self.lstm(seq)
        last_out = out[-1]
        if self.dr_rate:
            last_out = self.dropout(last_out)
        fc_out = self.fc(last_out)
        if self.dr_rate:
            fc_out = self.dropout(fc_out)
        return self.classifier(fc_out)


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc


if __name__ == '__main__':
    # Check arg validity
    logger.info('Starting...')
    if args.fine_tune:
        save_dir = Path(args.fine_tune_save).parent
        if not os.path.isdir(save_dir):
            raise ValueError(
                f"Check the path in --fine_tune_save option. Directory does not exist: {save_dir}")

    if args.robert or args.tobert:
        save_dir = Path(args.clf_save).parent
        if not os.path.isdir(save_dir):
            raise ValueError(
                f"Check the path in --fine_tune_save option. Directory does not exist: {save_dir}")

    logger.info(f"Args: device={args.device}, test_run={args.test_run}, "
                f"fine_tune={args.fine_tune}, fine_tune_save={args.fine_tune_save}, "
                 f"robert={args.robert}, clf_save={args.clf_save}")

    input_size = 768
    segment_len = 200
    overlap = 50
    batch_size = 16
    warmup_ratio = 0.1
    num_epochs_fine_tune = 1
    num_epochs_lstm = 1
    max_grad_norm = 1
    log_interval = 10
    learning_rate = 5e-5

    # Load model
    logger.info("Loading KoBERT...")
    bertmodel, vocab = get_pytorch_kobert_model()
    logger.info("Successfully loaded KoBERT.")

    # Load data
    data_path = str(PROJECT_DIR / 'data' /'processed' / '201003_labelled_{}.csv')
    logger.info(f"Loading data at {data_path}")

    if args.test_run:
        n_train_discard = 132
        n_test_discard = 58
    else:
        n_train_discard = n_test_discard = 1

    dataset_train = nlp.data.TSVDataset(data_path.format('train'), field_indices=[2, 3], num_discard_samples=n_train_discard)
    dataset_test = nlp.data.TSVDataset(data_path.format('test'), field_indices=[2, 3], num_discard_samples=n_test_discard)

    # Tokenizer
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    robert_data_train = SegmentedArticlesDataset(dataset_train, tok, segment_len, overlap, True, False)
    robert_data_test = SegmentedArticlesDataset(dataset_test, tok, segment_len, overlap, True, False)
    logger.info("Successfully loaded data. Articles are segmented and tokenized.")

    # Set device
    logger.info(f"Set device to {args.device}")
    device = torch.device(args.device)

    # 1. Fine-tune BERT ################################################################################################
    if args.fine_tune:
        logger.info("Fine-tuning KoBERT on data!")
        # 1.1 Load data
        data_train = BERTDataset.create_from_segmented(robert_data_train)
        data_test = BERTDataset.create_from_segmented(robert_data_test)

        train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=4)
        logger.info("Created data for KoBERT fine-tuning.")

        # 1.2 Set up classifier model.
        clf_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
        logger.info("KoBERT Classifier is instantiated.")

        # 1.3 Set up training parameters
        #       Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in clf_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in clf_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        t_total = len(train_dataloader) * num_epochs_fine_tune
        warmup_step = int(t_total * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        # 1.4 TRAIN!!!
        logger.info("Begin training")
        for e in range(num_epochs_fine_tune):
            train_acc = 0.0
            test_acc = 0.0
            clf_model.train()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length  # ㅁㅝ지?
                label = label.long().to(device)
                out = clf_model(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clf_model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                train_acc += calc_accuracy(out, label)
                if batch_id % log_interval == 0:
                    logger.info("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                             train_acc / (batch_id + 1)))
                    torch.save(clf_model.state_dict, args.fine_tune_save)
                    torch.save(optimizer.state_dict, args.fine_tune_save.split('.')[0] + '_optimizer.dict')

            logger.info("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
            clf_model.eval()
            for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                token_ids = token_ids.long().to(device)
                segment_ids = segment_ids.long().to(device)
                valid_length = valid_length
                label = label.long().to(device)
                out = clf_model(token_ids, valid_length, segment_ids)
                test_acc += calc_accuracy(out, label)
            logger.info("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

        torch.save(clf_model.state_dict, args.fine_tune_save)

    # 2. Train Recurrence over BERT ####################################################################################
    if args.robert:
        logger.info("Train RoBERT...")
        # 2.1 Set up parameters
        input_size = 768
        max_num_tokens = 9000
        segment_len = 200
        overlap = 50
        batch_size = 16
        warmup_ratio = 0.1
        num_epochs_fine_tune = 1
        num_epochs_lstm = 1
        max_grad_norm = 1
        log_interval = 200
        learning_rate = 1e-3
        lstm_hidden_size = 100
        fc_hidden_size = 30

        # 2.2 Instantiate model
        robert_model = RoBERT(
            input_size=input_size,
            lstm_hidden_size=lstm_hidden_size,
            fc_hidden_size=fc_hidden_size,
            dr_rate=0.5
        ).to(device)
        logger.info("RoBERT is instantiated.")

        # 2.3 Set up training parameters
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in robert_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in robert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # 2.4 Load BERT model
        if not args.fine_tune:
            if args.fine_tune_load != '':
                bertmodel = get_kobert_model(args.fine_tune_load)
            clf_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

        train_sequences = BERTOutputSequence(
            robert_data_train, batch_size=batch_size, bert_clf=clf_model, device=device)
        test_sequences = BERTOutputSequence(
            robert_data_test, batch_size=batch_size, bert_clf=clf_model, device=device)

        # TODO: use collate_fn argument in DataLoader to utilize multiprocessing etc?
        robert_train_dataloader = train_sequences
        robert_test_dataloader = test_sequences
        # train_dataloader = torch.utils.data.DataLoader(train_sequences, collate_fn=)

        t_total = len(robert_train_dataloader) * num_epochs_fine_tune
        warmup_step = int(t_total * warmup_ratio)

        # TODO: schedule according to the paper
        #  (initially 0.001, reduced by 0.95 if validation loss does not decrease for 3 epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=3)

        # 2.5 TRAIN!!!
        logger.info("Begin training")
        for e in range(num_epochs_lstm):
            train_acc = 0.0
            test_acc = 0.0
            robert_model.train()
            for batch_id, (articles_seq, label) in enumerate(
                    tqdm(robert_train_dataloader)):
                optimizer.zero_grad()

                label = label.long().to(device)
                lstm_out = robert_model(articles_seq)
                loss = loss_fn(lstm_out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(robert_model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step(loss)  # on validation loss?

                test_acc += calc_accuracy(lstm_out, label)
                if batch_id % log_interval == 0:
                    logger.info("epoch {} batch id {} loss {} train acc {}".format(e + 1, batch_id + 1, loss.data.cpu().numpy(),
                                                                             train_acc / (batch_id + 1)))

                    torch.save(robert_model.state_dict, args.clf_save)
                    torch.save(optimizer.state_dict, args.clf_save.split('.')[0] + '_optimizer.dict')

            logger.info("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))
            robert_model.eval()
            for batch_id, (articles_seq, label) in enumerate(
                    tqdm(robert_test_dataloader)):
                label = label.long().to(device)
                lstm_out = robert_model(articles_seq)
                test_acc += calc_accuracy(lstm_out, label)
            logger.info("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

        # 2.6 Save
        torch.save(robert_model.state_dict(), args.clf_save)
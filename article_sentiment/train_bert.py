#!/usr/bin/env python
# coding: utf-8
import os
import sys
from pathlib import Path
import argparse
import logging

import numpy as np
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
from article_sentiment.model import BERTClassifier
from article_sentiment.utils import calc_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('--device', help="`cpu` vs `gpu`", choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--fine_tune_save', help="save path for fine-tuned BERT classifier",
                    default=PROJECT_DIR / 'models' / 'bert_fine_tuned.dict', type=str)
parser.add_argument('--test_run', help="test run the code on small sample (2 lines of train and test each)", action='store_true')
# parser.add_argument('--save_log', help="wehther to save log or not", action="store_true")
parser.add_argument('-v', '--verbose', help="verbosity of log", action="store_true")
parser.add_argument('--seed', help="random seed for pytorch", default=0, type=int)
parser.add_argument('--warm_start', help='load saved fine-tuned clf and optimizer', action='store_true')
parser.add_argument('--validate', help="don't train, just validate", action='store_true')
args = parser.parse_args()

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    level=logging.DEBUG if args.verbose else logging.INFO)
logger = logging.getLogger('train_bert.py')
# log_format = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
#
# sh = logging.StreamHandler()
# sh.setLevel(logging.DEBUG if args.verbose else logging.INFO)
# sh.setFormatter(log_format)
# logger.addHandler(sh)

# if args.save_log:
#     fh = logging.FileHandler(
#         filename=PROJECT_DIR / 'logs' / f'train_bert_{datetime.now()}.log',
#         mode='w')
#     fh.setLevel(logging.DEBUG)
#     fh.setFormatter(log_format)
#     logger.addHandler(fh)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


if __name__ == '__main__':
    # Check arg validity
    logger.info('Starting train_bert.py...')

    save_dir = Path(args.fine_tune_save).parent
    if not os.path.isdir(save_dir):
        raise ValueError(
            f"Check the path in --fine_tune_save option. Directory does not exist: {save_dir}")

    if args.warm_start:
        assert os.path.exists(args.fine_tune_save), f"no state dict saved to warm-start at {args.fine_tune_save}"

    logger.info(f"Args: device={args.device}, test_run={args.test_run}, "
                f"fine_tune_save={args.fine_tune_save},")

    num_workers = os.cpu_count()
    input_size = 768
    segment_len = 200
    overlap = 50
    batch_size = 10
    warmup_ratio = 0.1
    num_epochs_fine_tune = 3
    max_grad_norm = 1
    log_interval = 1
    learning_rate = 5e-5

    # Load model
    logger.info("Loading KoBERT...")
    bertmodel, vocab = get_pytorch_kobert_model()
    logger.info("Successfully loaded KoBERT.")
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load data
    data_path = str(PROJECT_DIR / 'data' /'processed' / 'labelled_{}.csv')
    logger.info(f"Loading data at {data_path}")

    if args.test_run:
        n_train_discard = 119
        n_val_discard = 39
        n_test_discard = 39
    else:
        n_train_discard = n_val_discard = n_test_discard = 1

    dataset_train = nlp.data.TSVDataset(
        data_path.format('train'), field_indices=[0, 1], num_discard_samples=n_train_discard)
    dataset_val = nlp.data.TSVDataset(
        data_path.format('val'), field_indices=[0, 1], num_discard_samples=n_val_discard)
    dataset_test = nlp.data.TSVDataset(
        data_path.format('test'), field_indices=[0, 1], num_discard_samples=n_test_discard)

    # Tokenizer
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    robert_data_train = SegmentedArticlesDataset(dataset_train, tok, segment_len, overlap, True, False)
    robert_data_val = SegmentedArticlesDataset(dataset_val, tok, segment_len, overlap, True, False)
    robert_data_test = SegmentedArticlesDataset(dataset_test, tok, segment_len, overlap, True, False)
    logger.info("Successfully loaded data. Articles are segmented and tokenized.")
    
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    # Set device
    logger.info(f"Set device to {args.device}")
    device = torch.device(args.device)

    # 1. Fine-tune BERT ################################################################################################

    logger.info("Fine-tuning KoBERT on data!")
    # 1.1 Load data
    data_train = BERTDataset.create_from_segmented(robert_data_train)
    data_val = BERTDataset.create_from_segmented(robert_data_val)
    data_test = BERTDataset.create_from_segmented(robert_data_test)

    train_dataloader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(
        data_val, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, num_workers=num_workers)
    logger.info("Created data for KoBERT fine-tuning.")

    # 1.2 Set up classifier model.
    clf_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    logger.info("KoBERT Classifier is instantiated.")
    if args.warm_start:
        logger.info("Warm start: loading saved state dict...")
        state_dict = torch.load(args.fine_tune_save)
        clf_model.load_state_dict(state_dict)

    # 1.3 Set up training parameters
    #       Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in clf_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in clf_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    if args.warm_start:
        state_dict = torch.load(str(args.fine_tune_save).split('.')[0] + '_optimizer.dict')
        optimizer.load_state_dict(state_dict)
    logger.debug("Loaded optimizer")
    logger.debug(torch.cuda.memory_summary())

    loss_fn = nn.CrossEntropyLoss()

    t_total = len(train_dataloader) * num_epochs_fine_tune
    warmup_step = int(t_total * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # 1.4 TRAIN!!!
    logger.info("Begin training")
    for e in range(num_epochs_fine_tune):
        # 1.4.1 TRAIN
        if not args.validate:
            train_acc = 0.0
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
                    logger.info("epoch {} batch id {} loss {} train acc {}".format(
                        e + 1, batch_id + 1, loss.data.cpu().numpy(), train_acc / (batch_id + 1)))
                if batch_id % log_interval * 10 == 0:
                    torch.save(clf_model.state_dict(), args.fine_tune_save)
                    torch.save(optimizer.state_dict(), str(args.fine_tune_save).split('.')[0] + '_optimizer.dict')

            logger.info("epoch {} train acc {}".format(e + 1, train_acc / (batch_id + 1)))

            if args.device == 'cuda':
                logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

            del label, out, token_ids, segment_ids

            if args.device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

        # 1.4.2. Validate
        clf_model.eval()
        val_acc = 0.0
        val_loss = 0.0
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(val_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = clf_model(token_ids, valid_length, segment_ids)
            val_loss += loss_fn(out, label)
            val_acc += calc_accuracy(out, label)

        logger.info("epoch {} val acc {}, loss {}".format(e + 1, val_acc / (batch_id + 1), val_loss / (batch_id + 1)))

        if args.validate:
            break

    torch.save(clf_model.state_dict(), args.fine_tune_save)
    clf_model.eval()
    test_loss = 0.0
    test_acc = 0.0
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)
        out = clf_model(token_ids, valid_length, segment_ids)
        test_loss += loss_fn(out, label)
        test_acc += calc_accuracy(out, label)

    logger.info("test acc {}, loss {}".format(test_acc / (batch_id + 1), test_loss / (batch_id + 1)))

    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    del clf_model.classifier, optimizer, scheduler, train_dataloader, val_dataloader, label, out, token_ids, segment_ids

    if args.device == 'cuda':
        torch.cuda.empty_cache()
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

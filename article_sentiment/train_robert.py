#!/usr/bin/env python
# coding: utf-8
import os
from pathlib import Path
import argparse
import logging
from datetime import datetime

import torch
from torch import nn
import gluonnlp as nlp
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tqdm import tqdm

from article_sentiment.env import PROJECT_DIR
from article_sentiment.utils import calc_accuracy
from article_sentiment.kobert.utils import get_tokenizer
from article_sentiment.kobert.pytorch_kobert import get_pytorch_kobert_model, get_kobert_model
from article_sentiment.data.utils import SegmentedArticlesDataset
from article_sentiment.data.article_loader import BERTOutputSequence
from article_sentiment.model import BERTClassifier, RoBERT

parser = argparse.ArgumentParser()
parser.add_argument('--device', help="`cpu` vs `gpu`", choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--fine_tune_load', help="load path for fine-tuned BERT classifier", default='', type=str)
parser.add_argument('--clf_save', help='path to which classifier is saved', default=PROJECT_DIR / 'models' / 'classifier.dict')
parser.add_argument('--test_run', help="test run the code on small sample (2 lines of train and test each)", action='store_true')
args = parser.parse_args()


logging.basicConfig(level=logging.DEBUG, #filename=PROJECT_DIR / 'logs' / f'train_{datetime.now()}.log', filemode='w',
                    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
logger = logging.getLogger('train_robert.py')
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    # Check arg validity
    logger.info('Starting train_robert.py...')

    load_path = Path(args.fine_tune_load)
    if not os.path.exists(load_path):
        raise ValueError(f"fine_tune_load path at {args.fine_tune_load} does not exist!")

    save_dir = Path(args.clf_save).parent
    if not os.path.isdir(save_dir):
        raise ValueError(
            f"Check the path in --fine_tune_save option. Directory does not exist: {save_dir}")

    logger.info(f"Args: device={args.device}, test_run={args.test_run}, "
                f"fine_tune_load={args.fine_tune_load}, clf_save={args.clf_save}")

    num_workers = os.cpu_count()
    input_size = 768
    segment_len = 200
    overlap = 50

    # Load model
    logger.info("Loading KoBERT...")
    bertmodel, vocab = get_pytorch_kobert_model()
    logger.info("Successfully loaded KoBERT.")
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

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
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    # Set device
    logger.info(f"Set device to {args.device}")
    device = torch.device(args.device)

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
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

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
    logger.info("Loading KoBERT...")
    bertmodel = get_kobert_model(args.fine_tune_load)
    clf_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    logger.info("Successfully loaded KoBERT")
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    train_sequences = BERTOutputSequence(
        robert_data_train, batch_size=batch_size, bert_clf=clf_model, device=device)
    test_sequences = BERTOutputSequence(
        robert_data_test, batch_size=batch_size, bert_clf=clf_model, device=device)

    # TODO: use collate_fn argument in DataLoader to utilize multiprocessing etc?
    robert_train_dataloader = train_sequences
    robert_test_dataloader = test_sequences
    logger.info("Successfully loaded RoBERT data")
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    # def collate(lot):
    #     # lot = list of tuple (article, label)
    #     articles = []
    #     labels = []
    #     for sample, label in lot:
    #         bert_outputs = []
    #         for token_ids, valid_length, segment_ids in sample:
    #             token_ids = torch.reshape(torch.Tensor(token_ids), (1, -1)).long().to(device)
    #             segment_ids = torch.reshape(torch.Tensor(segment_ids), (1, -1)).long().to(device)
    #             valid_length = valid_length.reshape(1, )
    #
    #             # Get BERT output (batch_size, 768)
    #             # TODO: token ids, etc. must have ndim=2 !!!
    #             attention_mask = clf_model.gen_attention_mask(token_ids, valid_length)
    #             _, pooler = clf_model.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
    #                                       attention_mask=attention_mask.float().to(token_ids.device))
    #             bert_outputs.append(pooler)
    #         bert_output_seq = torch.cat(bert_outputs)
    #         articles.append(bert_output_seq)
    #         labels.append(label)
    #     articles = torch.nn.utils.rnn.pad_sequence(articles)
    #     labels = torch.Tensor(labels)
    #     return articles.float(), labels.long()
    #
    # robert_train_dataloader = torch.utils.data.DataLoader(robert_data_train, collate_fn=collate, batch_size=batch_size, num_workers=num_workers)
    # robert_test_dataloader = torch.utils.data.DataLoader(robert_data_test, collate_fn=collate, batch_size=batch_size, num_workers=num_workers)

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
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

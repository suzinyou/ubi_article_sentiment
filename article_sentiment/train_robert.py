#!/usr/bin/env python
# coding: utf-8
import argparse
import logging
import os
from pathlib import Path

import gluonnlp as nlp
import numpy as np
import torch
import wandb
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch import nn
from tqdm import tqdm
from transformers import AdamW

from article_sentiment.data.article_loader import BERTOutputSequence
from article_sentiment.data.utils import SegmentedArticlesDataset
from article_sentiment.env import PROJECT_DIR
from article_sentiment.kobert.pytorch_kobert import get_pytorch_kobert_model
from article_sentiment.kobert.utils import get_tokenizer
from article_sentiment.model import BERTClassifier, RoBERT
from article_sentiment.utils import num_correct

parser = argparse.ArgumentParser()
parser.add_argument('--device', help="`cpu` vs `gpu`", choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--fine_tune_load', help="load path for fine-tuned BERT classifier", default=PROJECT_DIR / 'models' / 'bert_fine_tuned.dict', type=str)
parser.add_argument('--clf_save', help='path to which classifier is saved', default=PROJECT_DIR / 'models' / 'classifier.dict')
parser.add_argument('--test_run', help="test run the code on small sample (2 lines of train and test each)", action='store_true')
parser.add_argument('--seed', help="random seed for pytorch", default=0, type=int)

parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-e', '--epochs', default=10, type=int)
args = parser.parse_args()


logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s')
logger = logging.getLogger('train_robert.py')


def train(model, device, train_loader, optimizer, epoch, classes):
    correct = 0
    cm = np.zeros((4, 4))
    model.train()

    for batch_id, (articles_seq, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        label = label.long().to(device)

        out = model(articles_seq)

        loss = nn.CrossEntropyLoss()(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        correct += num_correct(out, label)
        pred = torch.max(out, 1)[1].data.cpu().numpy()
        cm += confusion_matrix(label.data.cpu().numpy(), pred, labels=[0, 1, 2, 3])

        accuracy = correct / (batch_id * train_loader.batch_size + len(label))
        if (batch_id + 1) % config.log_interval == 0:
            logger.info(
                f"epoch {epoch + 1:2d} batch id {batch_id + 1:3d} "
                f"loss {loss.data.cpu().numpy():5f} train acc {accuracy:.5f}")
            logger.info(
                "Confusion matrix\n" +
                "True\\Pred " + ' '.join([f"{cat:>10}" for cat in classes]) + "\n" +
                '\n'.join([f"{cat:>10} " + ' '.join([f"{int(cnt):10d}" for cnt in row]) for cat, row in zip(classes, cm)])
            )

        if (batch_id + 1) % config.log_interval * 10 == 0:
            torch.save(model.state_dict(), args.clf_save)
            torch.save(optimizer.state_dict(), str(args.clf_save).split('.')[0] + '_optimizer.dict')

    wandb.log({
        # "Examples": example_images,
        "Train Accuracy": 100. * correct / len(train_loader.dataset),
        "Train Loss": loss,
        "Train Confusion Matrix": ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot().figure_})


def test(model, device, test_loader, scheduler, classes, epoch=None, mode='val'):
    # example_images = []

    model.eval()
    cm = np.zeros((4, 4))
    val_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_id, (articles_seq, label) in enumerate(tqdm(test_loader)):
            label = label.long().to(device)

            out = model(articles_seq)
            val_loss += nn.CrossEntropyLoss()(out, label).item()
            correct += num_correct(out, label)
            pred = torch.max(out, 1)[1].data.cpu().numpy()
            cm += confusion_matrix(label.data.cpu().numpy(), pred, labels=[0, 1, 2, 3])

    scheduler.step(val_loss)

    accuracy = 100. * correct / len(test_loader.dataset)
    wandb.log({
        # "Examples": example_images,
        f"{mode} Accuracy": accuracy,
        f"{mode} Loss": val_loss,
        f"{mode} Confusion Matrix": ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes).plot().figure_})

    if mode == 'val' or mode == 'Validation':
        logger.info(f"epoch {epoch + 1:2d} val acc {accuracy:.4f}, loss {val_loss:.5f}")        
    elif mode == 'test' or mode == 'Test':
        logger.info(f"test acc {accuracy:.4f}, loss {test_loss:.5f}")
    logger.info(
        "Confusion matrix\n" +
        "True\\Pred " + ' '.join([f"{cat:>10}" for cat in classes]) + "\n" +
        '\n'.join([f"{cat:>10} " + ' '.join([f"{int(cnt):10d}" for cnt in row]) for cat, row in zip(classes, cm)])
    )

if __name__ == '__main__':
    wandb.init(project="ubi_article_sentiment-RoBERT")
    # Check arg validity
    logger.info('Starting train_robert.py...')

    load_path = Path(args.fine_tune_load)
    if not os.path.exists(load_path):
        raise ValueError(f"fine_tune_load path at {args.fine_tune_load} does not exist!")

    save_dir = Path(args.clf_save).parent
    if not os.path.isdir(save_dir):
        raise ValueError(
            f"Check the path in --clf_save option. Directory does not exist: {save_dir}")

    logger.info(f"Args: device={args.device}, test_run={args.test_run}, "
                f"fine_tune_load={args.fine_tune_load}, clf_save={args.clf_save}")

    num_workers = os.cpu_count()
    input_size = 768

    config = wandb.config
    config.segment_len = 320
    config.overlap = 80
    config.batch_size = args.batch_size
    config.warmup_ratio = 0.1
    config.epochs = args.epochs
    config.max_grad_norm = 1
    config.log_interval = 3
    config.learning_rate = 1e-4
    config.lstm_hidden_size = 100
    config.fc_hidden_size = 30
    config.seed = args.seed

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load model ###############################################################################################
    logger.info("Loading KoBERT...")
    bertmodel, vocab = get_pytorch_kobert_model()
    logger.info("Successfully loaded KoBERT.")
    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    # Load data ###############################################################################################
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

    robert_data_train = SegmentedArticlesDataset(dataset_train, tok, config.segment_len, config.overlap, True, False)
    robert_data_val = SegmentedArticlesDataset(dataset_val, tok, config.segment_len, config.overlap, True, False)
    robert_data_test = SegmentedArticlesDataset(dataset_test, tok, config.segment_len, config.overlap, True, False)
    logger.info("Successfully loaded data. Articles are segmented and tokenized.")

    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    # Set device
    logger.info(f"Set device to {args.device}")
    device = torch.device(args.device)

    logger.info("Train RoBERT...")

    # 2.4 Load BERT model
    logger.info("Loading KoBERT...")
    clf_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    state_dict = torch.load(args.fine_tune_load)
    clf_model.load_state_dict(state_dict)
    logger.info("Successfully loaded KoBERT")

    if args.device == 'cuda':
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    train_sequences = BERTOutputSequence(
        robert_data_train, batch_size=config.batch_size, bert_clf=clf_model, device=device)
    val_sequences = BERTOutputSequence(
        robert_data_val, batch_size=config.batch_size, bert_clf=clf_model, device=device)
    test_sequences = BERTOutputSequence(
        robert_data_test, batch_size=config.batch_size, bert_clf=clf_model, device=device)

    # TODO: use collate_fn argument in DataLoader to utilize multiprocessing etc?
    robert_train_dataloader = train_sequences
    robert_val_dataloader = val_sequences
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

    # 2.2 Instantiate model
    robert_model = RoBERT(
        input_size=input_size,
        lstm_hidden_size=config.lstm_hidden_size,
        fc_hidden_size=config.fc_hidden_size,
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
        {'params': [p for n, p in robert_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # TODO: schedule according to the paper
    #  (initially 0.001, reduced by 0.95 if validation loss does not decrease for 3 epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.95, patience=3)

    # 2.5 TRAIN!!!
    logger.info("Begin training")
    for e in range(config.epochs):
        train(
            robert_model, device, robert_train_dataloader, optimizer,
            epoch=e, classes=robert_data_train.label_decoder
        )
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

        test(
            robert_model, device, robert_val_dataloader, scheduler,
            epoch=e, classes=robert_data_val.label_decoder, mode='Validation'
        )
        if args.device == 'cuda':
            torch.cuda.empty_cache()
            logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

    # 2.6 Save
    logger.info("Saving final RoBERT classifier...")
    torch.save(robert_model.state_dict(), args.clf_save)
    wandb.save(str(args.clf_save))

    # 2.7 Evaludate on test set
    test(
        robert_model, device, robert_val_dataloader, scheduler,
        classes=robert_data_val.label_decoder, mode='Test')

    if args.device == 'cuda':
        torch.cuda.empty_cache()
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

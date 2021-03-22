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
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumBarunGothic'

from article_sentiment.data.dataset import SegmentedArticlesDataset, BERTDataset
from article_sentiment.env import PROJECT_DIR
from article_sentiment.kobert.pytorch_kobert import get_pytorch_kobert_model
from article_sentiment.kobert.utils import get_tokenizer
from article_sentiment.model import BERTClassifier
from article_sentiment.utils import num_correct

parser = argparse.ArgumentParser()
parser.add_argument('--device', help="`cpu` vs `gpu`", choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--fine_tune_save', help="save path for fine-tuned BERT classifier",
                    default=PROJECT_DIR / 'models' / 'bert_fine_tuned.dict', type=str)
parser.add_argument('--test_run', help="test run the code on small sample (2 lines of train and test each)",
                    action='store_true')
parser.add_argument('--seed', help="random seed for pytorch", default=0, type=int)
parser.add_argument('--warm_start', help='load saved fine-tuned clf and optimizer', action='store_true')
parser.add_argument('-m', '--mode', help="train mode? val mode? all(both)?", choices=['train', 'validate', 'all'],
                    default='all')
parser.add_argument('-b', '--batch_size', default=16, type=int)
parser.add_argument('-e', '--epochs', default=3, type=int)
args = parser.parse_args()

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    level=logging.INFO)
logger = logging.getLogger('train_bert.py')


def train(model, device, train_loader, optimizer, scheduler, epoch, classes):
    correct = 0
    cm = np.zeros((4, 4))
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        loss = nn.CrossEntropyLoss()(out, label)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
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
                '\n'.join(
                    [f"{cat:>10} " + ' '.join([f"{int(cnt):10d}" for cnt in row]) for cat, row in zip(classes, cm)])
            )

        if (batch_id + 1) % config.log_interval * 10 == 0:
            torch.save(model.state_dict(), args.fine_tune_save)
            torch.save(optimizer.state_dict(), str(args.fine_tune_save).split('.')[0] + '_optimizer.dict')

    return {
        # "Examples": example_images,
        "Train Accuracy": 100. * correct / len(train_loader.dataset),
        "Train Loss": loss,
        "Train Confusion Matrix": ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes
        ).plot().figure_
    }


def test(model, device, test_loader, classes, epoch=None, mode='val'):
    # example_images = []

    model.eval()
    cm = np.zeros((4, 4))
    val_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_loader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
            val_loss += nn.CrossEntropyLoss()(out, label).item()
            correct += num_correct(out, label)
            pred = torch.max(out, 1)[1].data.cpu().numpy()
            cm += confusion_matrix(label.data.cpu().numpy(), pred, labels=[0, 1, 2, 3])

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info(f"epoch {epoch + 1:2d} val acc {accuracy:.4f}, loss {val_loss:.5f}")
    logger.info(
        "Confusion matrix\n" +
        "True\\Pred " + ' '.join([f"{cat:>10}" for cat in classes]) + "\n" +
        '\n'.join([f"{cat:>10} " + ' '.join([f"{int(cnt):10d}" for cnt in row]) for cat, row in zip(classes, cm)])
    )

    return {
        # "Examples": example_images,
        f"{mode} Accuracy": accuracy,
        f"{mode} Loss": val_loss,
        f"{mode} Confusion Matrix": ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes
        ).plot().figure_
    }


if __name__ == '__main__':
    wandb.init(project="ubi_article_sentiment")
     
    logger.info('Starting train_bert.py...')

    # Check arg validity
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

    config = wandb.config
    config.segment_len = 200
    config.overlap = 50
    config.batch_size = args.batch_size
    config.warmup_ratio = 0.1
    config.epochs = args.epochs
    config.max_grad_norm = 1
    config.log_interval = 3
    config.learning_rate = 5e-5
    config.dropout_rate = 0.0
    config.seed = args.seed

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load KoBERT ###############################################################################################
    logger.info("Loading KoBERT...")
    bertmodel, vocab = get_pytorch_kobert_model()
    logger.info("Successfully loaded KoBERT.")

    # Load data ###############################################################################################
    data_path = str(PROJECT_DIR / 'data' / 'processed' / 'labelled320_{}.csv')
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

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    robert_data_train = SegmentedArticlesDataset(
        dataset=dataset_train, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)
    robert_data_val = SegmentedArticlesDataset(
        dataset=dataset_val, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)
    robert_data_test = SegmentedArticlesDataset(
        dataset=dataset_test, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)
    logger.info("Successfully loaded data. Articles are segmented and tokenized.")

    # Set device #######################################################################################################
    logger.info(f"Set device to {args.device}")
    device = torch.device(args.device)

    if args.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 1. Fine-tune BERT ################################################################################################
    logger.info("Fine-tuning KoBERT on data!")

    # 1.1 Create DataLoader (1 sample = 1 segment)
    data_train = BERTDataset.create_from_segmented(robert_data_train)
    data_val = BERTDataset.create_from_segmented(robert_data_val)
    data_test = BERTDataset.create_from_segmented(robert_data_test)

    train_dataloader = DataLoader(
        data_train, batch_size=config.batch_size,
        sampler=WeightedRandomSampler(data_train.sample_weight, len(data_train)),
        num_workers=num_workers)
    val_dataloader = DataLoader(data_val, batch_size=config.batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(data_test, batch_size=config.batch_size, num_workers=num_workers)
    logger.info("Created data for KoBERT fine-tuning.")

    # 1.2 Set up classifier model.
    clf_model = BERTClassifier(bertmodel, dr_rate=config.dropout_rate).to(device)
    if args.warm_start or args.mode == 'validate':
        logger.info("Loading saved state dict...")
        state_dict = torch.load(args.fine_tune_save)
        clf_model.load_state_dict(state_dict)
        logger.debug("Loaded saved state dict to BERTClassifier.")

    if args.device == 'cuda':
        logger.debug(torch.cuda.memory_summary())

    # 1.3 Set up training parameters
    #       Prepare optimizer and schedule (linear warmup and decay)
    loss_fn = nn.CrossEntropyLoss()

    if args.mode in ('train', 'all'):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in clf_model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in clf_model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)

        if args.warm_start:
            state_dict = torch.load(str(args.fine_tune_save).split('.')[0] + '_optimizer.dict')
            optimizer.load_state_dict(state_dict)

        logger.debug("Loaded optimizer")

        if args.device == 'cuda':
            logger.debug(torch.cuda.memory_summary())

        t_total = len(train_dataloader) * config.epochs
        warmup_step = int(t_total * config.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    # 1.4 TRAIN!!!
    logger.info("Begin training")
    wandb.watch(clf_model, log="all")
    for e in range(config.epochs):
        # 1.4.1 TRAIN
        logs = {}
        if args.mode in ('train', 'all'):
            train_logs = train(
                clf_model, device, train_dataloader, optimizer, scheduler, epoch=e,
                classes=robert_data_train.label_decoder)
            logs.update(train_logs)
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

        # 1.4.2. Validate
        if args.mode in ('validate', 'all'):
            test_logs = test(
                clf_model, device, val_dataloader,
                classes=robert_data_val.label_decoder, epoch=e, mode='Validation'
            )
            logs.update(test_logs)

            if args.device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

            if args.mode == 'validate':
                break

        wandb.log(logs)

    logger.info("Saving final BERTClassifier state dict...")
    torch.save(clf_model.state_dict(), args.fine_tune_save)
    wandb.save(str(args.fine_tune_save))

    # Evaluate on test set
    test(clf_model, device, test_dataloader, classes=robert_data_test.label_decoder, mode='Test')

    if args.device == 'cuda':
        torch.cuda.empty_cache()
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

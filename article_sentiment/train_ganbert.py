#!/usr/bin/env python
# coding: utf-8
from article_sentiment.data.dataset import SegmentedArticlesDataset, BERTDataset
from article_sentiment.env import PROJECT_DIR, LOG_DIR
from article_sentiment.kobert.pytorch_kobert import get_pytorch_kobert_model
from article_sentiment.kobert.utils import get_tokenizer
from article_sentiment.model import Discriminator, Generator
from article_sentiment.model.loss import LossGenerator, LossDiscriminator
from article_sentiment.utils import num_correct

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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumBarunGothic'


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
parser.add_argument('--wandb_off', help="turn off wandb", action='store_true')
args = parser.parse_args()

logging.basicConfig(
    format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
    level=logging.INFO)
logger = logging.getLogger('train_bert.py')


def run(bert,
        discriminator, generator,
        # discriminator_loss, generator_loss,
        device, data_loader, classes, mode,
        optimizer_d=None, optimizer_g=None, scheduler=None, epoch=None):
    """
    :param optimizer_d: required if mode == 'train
    """
    assert mode in ('train', 'validation', 'test', 'predict')
    if mode == 'train':
        assert optimizer_d is not None
        assert optimizer_g is not None
        assert scheduler is not None
        assert epoch is not None

    correct = 0
    loss_d_total = 0
    loss_g_total = 0
    cm = np.zeros((len(classes), len(classes)))

    if mode == 'train':
        bert.train()
        discriminator.train()
        generator.train()
    else:
        bert.eval()
        discriminator.eval()
        generator.eval()

    # TODO: train_loader must load is_labeled_mask!!!!!!!!
    for batch_id, batch_data in enumerate(tqdm(data_loader)):
        token_ids, valid_length, segment_ids, label, is_labeled_mask = batch_data

        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length = valid_length
        label = label.long().to(device)

        if mode == 'train':
            optimizer_d.zero_grad()  # TODO: call zero_grad on optimizer or model?
            optimizer_g.zero_grad()

        # create attention mask for BERT
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        attention_mask = attention_mask.float().to(device)

        # get BERT output
        _, pooler = model_bert(
            input_ids=token_ids,
            token_type_ids=segment_ids,
            attention_mask=attention_mask
        )

        # pass thru discriminator
        D_real_features, D_real_logits, D_real_probs = discriminator(pooler)

        # pass noise thru generator
        # TODO: pass config as param for this function?
        z = torch.rand((config.batch_size, config.dim_latent_z)).to(device)
        x_g = generator(z)
        D_fake_features, D_fake_logits, D_fake_probs = discriminator(x_g)

        # Loss
        loss_d, clf_probs = LossDiscriminator(
            num_classes=len(classes), is_training=mode == 'train'
        )(D_real_logits, D_real_probs, D_fake_probs.detach(), label, is_labeled_mask)
        loss_g = LossGenerator()(
            D_fake_probs, D_fake_features, D_real_features.detach())

        loss_d_total += loss_d.data.cpu().numpy()
        loss_g_total += loss_g.data.cpu().numpy()

        if mode == 'train':
            # Backprop
            loss_d.backward()
            # TODO: clip grad norms only for bert??
            torch.nn.utils.clip_grad_norm_(model_bert.parameters(), config.max_grad_norm)
            optimizer_d.step()
            # TODO: when to do scheduler.step()?
            scheduler.step()  # Update learning rate schedule

            loss_g.backward()
            optimizer_g.step()

        # Evaluate
        eval_probs = clf_probs[is_labeled_mask]
        correct += num_correct(eval_probs, label[is_labeled_mask])  # TODO: check if per_ex_lossis
        pred = torch.max(eval_probs, 1)[1].data.cpu().numpy()
        cm += confusion_matrix(
            label[is_labeled_mask], pred, labels=list(range(len(classes))))
        accuracy = correct / (batch_id * data_loader.batch_size + is_labeled_mask.sum().data.cpu().numpy())

        if (batch_id + 1) % config.log_interval == 0:
            logger.info(
                f"epoch {epoch + 1:2d} batch id {batch_id + 1:3d} "
                f"loss_d {loss_d.data.cpu().numpy()[0]:5f} loss_g {loss_g.data.cpu().numpy():.5f} train acc {accuracy:.5f}")
            logger.info(
                "Confusion matrix\n" +
                "True\\Pred " + ' '.join([f"{cat:>10}" for cat in classes]) + "\n" +
                '\n'.join(
                    [f"{cat:>10} " + ' '.join([f"{int(cnt):10d}" for cnt in row]) for cat, row in zip(classes, cm)])
            )

        if (batch_id + 1) % config.log_interval * 10 == 0:
            # for model in (bertmodel, discriminator, generator):
            torch.save({
                'bert': bert.state_dict(),
                'discriminator': discriminator.state_dict(),
                'generator': generator.state_dict()
            }, args.fine_tune_save)
            torch.save({
                'bert-discriminator': optimizer_d.state_dict(),
                'generator': optimizer_g.state_dict()
            }, str(args.fine_tune_save).split('.')[0] + '_optimizer.dict')

    accuracy = correct / len(data_loader.dataset)
    # tb.add_scalar('Loss_D', loss_d_total, epoch)
    # tb.add_scalar('Loss_G', loss_g_total, epoch)
    # tb.add_scalar('Accuracy', accuracy, epoch)
    #
    # tb.add_histogram('discriminator.main[-1].bias', discriminator.main[-1].bias, epoch)
    # tb.add_histogram('discriminator.main[-1].weight', discriminator.main[-1].weight, epoch)
    # tb.add_histogram('discriminator.main[-1].weight.grad', discriminator.main[-1].weight.grad, epoch)
    # tb.add_histogram('generator.main[-1].bias', generator.main[-1].bias, epoch)
    # tb.add_histogram('generator.main[-1].weight', generator.main[-1].weight, epoch)
    # tb.add_histogram('generator.main[-1].weight.grad', generator.main[-1].weight.grad, epoch)

    return {
        # "Examples": example_images,
        f" Accuracy ({mode})": accuracy,
        f" Loss_D ({mode})": loss_d_total,
        f" Loss_G ({mode})": loss_g_total,
        f" Confusion Matrix ({mode})": ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=classes
        ).plot().figure_
    }


# def test(bert, discriminator, generator, device, test_loader, classes, epoch=None, mode='val'):
#     # example_images = []
#
#     bert.eval()
#     discriminator.eval()
#     generator.eval()
#
#     cm = np.zeros((len(classes), len(classes)))
#     val_loss = 0.0
#     correct = 0.0
#     with torch.no_grad():
#         for batch_id, (token_ids, valid_length, segment_ids, label, is_labeled_mask) in enumerate(tqdm(test_loader)):
#             token_ids = token_ids.long().to(device)
#             segment_ids = segment_ids.long().to(device)
#             valid_length = valid_length
#             label = label.long().to(device)
#
#             # create attention mask for BERT
#             attention_mask = torch.zeros_like(token_ids)
#             for i, v in enumerate(valid_length):
#                 attention_mask[i][:v] = 1
#             attention_mask = attention_mask.float().to(device)
#
#             # get BERT output
#             _, pooler = model_bert(
#                 input_ids=token_ids,
#                 token_type_ids=segment_ids,
#                 attention_mask=attention_mask
#             )
#
#             # pass thru discriminator
#             D_real_features, D_real_logits, D_real_probs = discriminator(pooler)
#
#             # pass noise thru generator
#             # TODO: pass config as param for this function?
#             z = torch.rand((config.batch_size, config.dim_latent_z))
#             x_g = model_G(z)
#             D_fake_features, D_fake_logits, D_fake_probs = discriminator(x_g)
#
#             val_loss += nn.CrossEntropyLoss()(out, label).item()
#             correct += num_correct(out, label)
#             pred = torch.max(out, 1)[1].data.cpu().numpy()
#             cm += confusion_matrix(label.data.cpu().numpy(), pred, labels=list(range(len(classes))))
#
#     accuracy = 100. * correct / len(test_loader.dataset)
#
#     logger.info(f"epoch {epoch + 1:2d} val acc {accuracy:.4f}, loss {val_loss:.5f}")
#     logger.info(
#         "Confusion matrix\n" +
#         "True\\Pred " + ' '.join([f"{cat:>10}" for cat in classes]) + "\n" +
#         '\n'.join([f"{cat:>10} " + ' '.join([f"{int(cnt):10d}" for cnt in row]) for cat, row in zip(classes, cm)])
#     )
#
#     return {
#         # "Examples": example_images,
#         f"{mode} Accuracy": accuracy,
#         f"{mode} Loss": val_loss,
#         f"{mode} Confusion Matrix": ConfusionMatrixDisplay(
#             confusion_matrix=cm, display_labels=classes
#         ).plot().figure_
#     }


if __name__ == '__main__':
    if not args.wandb_off:
        wandb_run = wandb.init(
            project="ubi_article_sentiment-GANBERT",
            dir=LOG_DIR,
            name='Testing for weights logging',)
    # tb_writer = SummaryWriter(LOG_DIR / 'tensorboard')

    logger.info('Starting train_ganbert.py...')

    # Check arg validity
    run_id = wandb.run.id
    run_log_dir = LOG_DIR / run_id
    run_log_dir.mkdir(exist_ok=True, parents=True)
    # save_dir = Path(args.fine_tune_save).parent
    # if not os.path.isdir(save_dir):
    #     raise ValueError(
    #         f"Check the path in --fine_tune_save option. Directory does not exist: {save_dir}")

    if args.warm_start:
        assert os.path.exists(args.fine_tune_save), f"no state dict saved to warm-start at {args.fine_tune_save}"

    logger.info(f"Args: device={args.device}, test_run={args.test_run}, "
                f"fine_tune_save={args.fine_tune_save},")

    num_workers = os.cpu_count()
    input_size = 768

    if args.wandb_off:
        class Config(object):
            def __init__(self):
                pass
        config = Config()
    else:
        config = wandb.config
    config.segment_len = 200
    config.overlap = 50
    config.batch_size = args.batch_size
    config.warmup_ratio = 0.1
    config.epochs = args.epochs
    config.max_grad_norm = 1
    config.log_interval = 10
    config.learning_rate = 5e-5
    config.dropout_rate = 0.01
    config.seed = args.seed
    config.filter_kw_segment = True
    config.dim_latent_z = 100

    # Set random seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load KoBERT ###############################################################################################
    logger.info("Loading KoBERT...")
    model_bert, vocab = get_pytorch_kobert_model()
    logger.info("Successfully loaded KoBERT.")

    # Load data ###############################################################################################
    data_path = str(PROJECT_DIR / 'data' / 'processed' / 'ganbert' / '{}.csv')
    logger.info(f"Loading data at {data_path}")

    if args.test_run:
        n_train_discard = 160 - 16
        n_val_discard = 80 - 16
        n_test_discard = 80 - 16
        n_labeled_discard = n_train_discard + n_val_discard + n_test_discard
        n_train_unlabeled_discard = 2071 - n_labeled_discard - 16*2
    else:
        n_train_discard = n_train_unlabeled_discard = n_val_discard = n_test_discard = 1

    # Read dataset files
    labeled_examples = nlp.data.TSVDataset(
        data_path.format('train_labeled'), field_indices=[0, 1], num_discard_samples=n_train_discard)
    unlabeled_examples = nlp.data.TSVDataset(
        data_path.format('train_unlabeled'), field_indices=[0, 1], num_discard_samples=n_train_unlabeled_discard)
    val_examples = nlp.data.TSVDataset(
        data_path.format('val_labeled'), field_indices=[0, 1], num_discard_samples=n_val_discard)
    test_examples = nlp.data.TSVDataset(
        data_path.format('test_labeled'), field_indices=[0, 1], num_discard_samples=n_val_discard)

    # Load tokenizer
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    # Segment articles into pre-defined segment lengths.
    train_labeled_articles = SegmentedArticlesDataset(
        dataset=labeled_examples, is_labeled=True, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)
    train_unlabeled_articles = SegmentedArticlesDataset(
        dataset=unlabeled_examples, is_labeled=False, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)
    val_articles = SegmentedArticlesDataset(
        dataset=val_examples, is_labeled=True, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)
    test_articles = SegmentedArticlesDataset(
        dataset=test_examples, is_labeled=True, bert_tokenizer=tok,
        seg_len=config.segment_len, shift=config.overlap,
        pad=True, pair=False, filter_kw_segment=config.filter_kw_segment)

    # Create BERTDataset - gives tuple
    #   (   token_ids,          shape: (args.batch_size, config.segment_length)
    #       valid_lengths,      shape: (args.batch_size,)
    #       input_token_ids,    shape: (args.batch_size, config.segment_length) - all 0s since it's not Q&A or NLI
    #       label,              shape: (args.batch_size,)
    #       is_labeled_mask)    shape: (args.batch_size,)
    bert_dataset_train = BERTDataset.create_from_segmented(train_labeled_articles, train_unlabeled_articles)
    bert_dataset_val = BERTDataset.create_from_segmented(val_articles, None)
    bert_dataset_test = BERTDataset.create_from_segmented(test_articles, None)

    logger.info("Successfully loaded data. Articles are segmented and tokenized.")

    # Set device #######################################################################################################
    logger.info(f"Set device to {args.device}")
    device = torch.device(args.device)

    if args.device == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # 1. Fine-tune BERT ################################################################################################
    logger.info("Fine-tuning KoBERT on data!")

    # 1.1 Create DataLoader (
    train_dataloader = DataLoader(
        bert_dataset_train, batch_size=config.batch_size, num_workers=num_workers,
        sampler=WeightedRandomSampler(bert_dataset_train.sample_weight, len(bert_dataset_train)),)
    val_dataloader = DataLoader(
        bert_dataset_val, batch_size=config.batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(
        bert_dataset_test, batch_size=config.batch_size, num_workers=num_workers)
    logger.info("Created data for KoBERT fine-tuning.")

    # 1.2 Set up GAN-BERT
    model_D = Discriminator(
        dim_bert_out=model_bert.config.hidden_size,
        dim_hidden=model_bert.config.hidden_size,  # ???
        num_hidden_layers=1,
        dr_rate=0.1, num_classes=4, n_gpu=0
    ).to(device)

    model_G = Generator(
        dim_latent_z=100, num_hidden_layers=1,
        dr_rate=0.1, num_classes=4, dim_hidden=768, n_gpu=0
    ).to(device)

    if args.warm_start or args.mode == 'validate':
        logger.info("Loading saved state dict...")
        checkpoint = torch.load(args.fine_tune_save)
        model_bert.load_state_dict(checkpoint['bert'])
        model_D.load_state_dict(checkpoint['discriminator'])
        model_G.load_state_dict(checkpoint['generator'])
        logger.debug("Loaded saved state dict to BERTClassifier.")

    if args.device == 'cuda':
        logger.debug(torch.cuda.memory_summary())

    # 1.3 Set up training parameters
    #       Prepare optimizer and schedule (linear warmup and decay)

    if args.mode in ('train', 'all'):
        no_decay = ['bias', 'LayerNorm.weight']
        bert_decay_params = [p for n, p in model_bert.named_parameters() if not any(nd in n for nd in no_decay)]
        bert_no_decay_params = [p for n, p in model_bert.named_parameters() if any(nd in n for nd in no_decay)]
        optimizer_D_grouped_parameters = [
            {'params': bert_decay_params + list(model_D.parameters()),
             'weight_decay': 0.01},
            {'params': bert_no_decay_params,
             'weight_decay': 0.0}
        ]

        optimizer_D = AdamW(optimizer_D_grouped_parameters, lr=config.learning_rate)
        optimizer_G = AdamW(model_G.parameters(), lr=config.learning_rate, weight_decay=0.01)

        if args.warm_start:
            checkpoint = torch.load(str(args.fine_tune_save).split('.')[0] + '_optimizer.dict')
            optimizer_D.load_state_dict(checkpoint['bert-discriminator'])
            optimizer_G.load_state_dict(checkpoint['generator'])

        logger.debug("Loaded optimizer")

        if args.device == 'cuda':
            logger.debug(torch.cuda.memory_summary())

        t_total = len(train_dataloader) * config.epochs
        warmup_step = int(t_total * config.warmup_ratio)

        scheduler_D = get_linear_schedule_with_warmup(
            optimizer_D, num_warmup_steps=warmup_step, num_training_steps=t_total)
        # TODO: Need a scheduler for G?

    # 1.4 TRAIN!!!
    logger.info("Begin training")
    if not args.wandb_off:
        wandb.watch(model_D, log="all")
        wandb.watch(model_G, log="all")

    # tb_writer.add_graph(model_bert)
    # tb_writer.add_graph(model_D)
    # tb_writer.add_graph(model_G)

    for e in range(config.epochs):
        # 1.4.1 TRAIN
        logs = {}
        if args.mode in ('train', 'all'):
            train_logs = run(
                model_bert, model_D, model_G, device, train_dataloader,
                optimizer_g=optimizer_G, optimizer_d=optimizer_D, scheduler=scheduler_D,
                epoch=e, classes=train_labeled_articles.label_decoder, mode='train')
            logs.update(train_logs)
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

        # 1.4.2. Validate
        if args.mode in ('validate', 'all'):
            val_logs = run(
                model_bert, model_D, model_G, device, val_dataloader,
                classes=val_articles.label_decoder, epoch=e, mode='validation'
            )
            logs.update(val_logs)

            if args.device == 'cuda':
                torch.cuda.empty_cache()
                logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

            if args.mode == 'validate':
                break

        if not args.wandb_off:
            wandb.log(logs)

        logger.info(f"Epoch {e:02d}: Saving state dicts...")
        torch.save({
            'bert': model_bert.state_dict(),
            'discriminator': model_D.state_dict(),
            'generator': model_G.state_dict()
        }, run_log_dir / f'models-{e:04d}.dict')  # TODO: use different checkpoint names for each epoch?
        torch.save({
            'bert-discriminator': optimizer_D.state_dict(),
            'generator': optimizer_G.state_dict()
        }, run_log_dir / f'optimizers-{e:04d}.dict')
        if not args.wandb_off:
            wandb.save(str(run_log_dir / f'models-{e:04d}.dict'))
            wandb.save(str(run_log_dir / f'optimizers-{e:04d}.dict'))

    # Evaluate on test set
    run(model_bert, model_D, model_G, device, test_dataloader,
        classes=test_articles.label_decoder, mode='test')

    if args.device == 'cuda':
        torch.cuda.empty_cache()
        logger.debug(f"Cuda memory summary: {torch.cuda.memory_summary()}")

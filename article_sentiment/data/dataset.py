import gluonnlp as nlp
import numpy as np
from torch.utils.data import Dataset

from article_sentiment.data.utils import (
    generate_overlapping_segments, recover_sentence_from_tokens,
    generate_valid_lengths
)


class SegmentedArticlesDataset(Dataset):
    def __init__(self, dataset, is_labeled, bert_tokenizer, seg_len, shift,
                 pad, pair, filter_kw_segment=False):
        self.is_labeled = is_labeled
        self.transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=seg_len+2, pad=pad, pair=pair)

        articles = []
        if is_labeled:
            labels = []

        for sample, label in dataset:
            # todo: treat cases where pad, pair are reverse
            tokens = bert_tokenizer(sample)
            segments = [
                recover_sentence_from_tokens(toks)
                for toks in generate_overlapping_segments(tokens, seg_len, shift)
            ]
            # input_token_ids, valid_length, input_token_types = self.transform(segments)
            # ids = list(generate_overlapping_segments(input_token_ids, seg_len, shift))
            # val_len = generate_valid_lengths(valid_length, seg_len, shift)
            # types = list(generate_overlapping_segments(input_token_types, seg_len, shift))

            # articles.append(list(zip(ids, val_len, types)))
            if filter_kw_segment:
                has_kw = []

                for segment in segments:
                    if '기본소득' in segment or '기본 소득' in segment:
                        has_kw.append(True)
                    else:
                        has_kw.append(False)
                has_kw = np.asarray(has_kw)
                before_has_kw = np.r_[has_kw[1:], False]
                after_has_kw = np.r_[False, has_kw[:-1]]
                around_kw = np.logical_or(np.logical_or(has_kw, before_has_kw), after_has_kw)

                valid_segments = [
                    s for s, is_around_kw in zip(segments, around_kw) if is_around_kw
                ]
            else:
                valid_segments = segments

            articles.append([self.transform([segment]) for segment in valid_segments])
            if is_labeled:
                labels.append(label)

        self.articles = articles
        if is_labeled:
            unique_labels = np.unique(labels)
            label_encoder = {l: j for j, l in enumerate(unique_labels)}

            self.labels = [np.int32(label_encoder[l]) for l in labels]
            self.label_encoder = label_encoder
            self.label_decoder = []
            for j in range(len(label_encoder)):
                for k, v in label_encoder.items():
                    if v == j:
                        self.label_decoder.append(k)

    def __getitem__(self, i):
        if self.is_labeled:
            return self.articles[i], self.labels[i]
        else:
            return self.articles[i]

    def __len__(self):
        return len(self.articles)


class BERTDataset(Dataset):
    # TODO: make this dataset use the result from SegmentedArticlesDataset
    def __init__(self, bert_inputs, labels, is_labeled_mask):
        self.bert_inputs = bert_inputs
        self.labels = np.asarray(labels)
        self.is_labeled_mask = np.asarray(is_labeled_mask)

    @classmethod
    def create_from_dataset(cls, labeled_dataset, unlabeled_dataset, bert_tokenizer, max_doc_len, pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_doc_len, pad=pad, pair=pair)

        article_segments = []
        labels = []
        is_labeled_mask = []

        for sample, label in labeled_dataset:
            article_segments.append(transform([sample]))
            labels.append(np.int32(label))
            is_labeled_mask.append(True)

        if unlabeled_dataset is not None:
            for sample, label in unlabeled_dataset:
                article_segments.append(transform([sample]))
                labels.append(np.int32(0))  # doesn't matter
                is_labeled_mask.append(False)

        unique_labels = np.unique(labels)
        label_encoder = {l: j for j, l in enumerate(unique_labels)}
        is_labeled_mask = np.asarray(is_labeled_mask)

        cls.label_encoder = label_encoder
        return cls(article_segments, [np.int32(label_encoder[l]) for l in labels], is_labeled_mask)

    @classmethod
    def create_from_segmented(cls, labeled_articles_dataset, unlabeled_articles_dataset):
        """create instance of dataset with (segment, label) pairs, with multiple segments from the same article having
        the same label"""
        assert isinstance(labeled_articles_dataset, SegmentedArticlesDataset)
        assert unlabeled_articles_dataset is None or isinstance(unlabeled_articles_dataset, SegmentedArticlesDataset)
        bert_inputs = []
        labels = []
        is_labeled = []

        for example in labeled_articles_dataset:
            list_of_segs, label = example
            n_segs = len(list_of_segs)
            for seg_bert_input in list_of_segs:
                bert_inputs.append(seg_bert_input)
            labels.extend([label] * n_segs)
            is_labeled.extend([True] * n_segs)

        if unlabeled_articles_dataset is not None:
            for list_of_segs in unlabeled_articles_dataset: # there is no label
                n_segs = len(list_of_segs)
                for seg_bert_input in list_of_segs:
                    bert_inputs.append(seg_bert_input)
                labels.extend([np.int32(0)] * n_segs)  # placeholders
                is_labeled.extend([False] * n_segs)

        return cls(bert_inputs, labels, is_labeled_mask=is_labeled)

    @property
    def sample_weight(self):
        num_samples = np.unique(self.labels, return_counts=True)[1]
        class_weights = 1 / num_samples
        sample_weight = class_weights[self.labels]
        return sample_weight

    def __getitem__(self, i):
        # TODO: make this compatible with base class
        return self.bert_inputs[i] + (self.labels[i], self.is_labeled_mask[i])

    def __len__(self):
        return len(self.bert_inputs)

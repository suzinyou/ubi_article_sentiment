import gluonnlp as nlp
import numpy as np
from torch.utils.data import Dataset

from article_sentiment.data.utils import (
    generate_overlapping_segments, recover_sentence_from_tokens,
    generate_valid_lengths
)


class SegmentedArticlesDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, seg_len, shift,
                 pad, pair):
        self.transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=seg_len+2, pad=pad, pair=pair)

        articles = []
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
            articles.append([self.transform([segment]) for segment in segments])
            labels.append(label)

        unique_labels = np.unique(labels)
        label_encoder = {l: j for j, l in enumerate(unique_labels)}

        self.articles = articles
        self.labels = [np.int32(label_encoder[l]) for l in labels]
        self.label_encoder = label_encoder
        self.label_decoder = []
        for j in range(len(label_encoder)):
            for k, v in label_encoder.items():
                if v == j:
                    self.label_decoder.append(k)

    def __getitem__(self, i):
        return (self.articles[i], self.labels[i],)

    def __len__(self):
        return (len(self.labels))


class BERTDataset(Dataset):
    # TODO: make this dataset use the result from SegmentedArticlesDataset
    def __init__(self, segments, labels):
        self.segments = segments
        self.labels = labels

    @classmethod
    def create_from_dataset(cls, dataset, bert_tokenizer, max_doc_len, seg_len, shift,
                 pad, pair):
        cls.transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_doc_len, pad=pad, pair=pair)

        article_segments = []
        labels = []

        for sample, label in dataset:
            # todo: treat cases where pad, pair are reverse
            input_token_ids, valid_length, input_token_types = cls.transform([sample])

            ids = list(generate_overlapping_segments(input_token_ids, seg_len, shift))
            val_len = generate_valid_lengths(valid_length, seg_len, shift)
            types = list(generate_overlapping_segments(input_token_types, seg_len, shift))

            article_segments.extend(list(zip(ids, val_len, types)))
            labels.extend([label] * len(val_len))

        unique_labels = np.unique(labels)
        label_encoder = {l: j for j, l in enumerate(unique_labels)}

        cls.label_encoder = label_encoder
        return cls(article_segments, [np.int32(label_encoder[l]) for l in labels])

    @classmethod
    def create_from_segmented(cls, articles_dataset):
        """create instance of dataset with (segment, label) pairs, with multiple segments from the same article having
        the same label"""
        segments = []
        labels = []

        for article, label in articles_dataset:
            for segment in article:
                segments.append(segment)
                labels.append(label)

        return cls(segments, labels)

    @property
    def sample_weight(self):
        num_samples = np.unique(self.labels, return_counts=True)[1]
        class_weights = 1 / num_samples
        sample_weight = class_weights[self.labels]
        return sample_weight

    def __getitem__(self, i):
        # TODO: make this compatible with base class
        return (self.segments[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))
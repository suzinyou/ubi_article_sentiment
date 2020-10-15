import re
import gluonnlp as nlp
import numpy as np
from torch.utils.data import Dataset


space = re.compile('▁')


def generate_overlapping_segments(array, length, overlap):
    array = np.array(array)
    i = 0
    while i < array.shape[-1] - overlap:
        if array.ndim > 1:
            segment = array[:, i:i + length]
        else:
            segment = array[i:i + length]
        i += length - overlap
        yield segment


def f(x, length, overlap):
    non_overlap = length - overlap
    n = x // non_overlap
    r = x % non_overlap
    if r == overlap:
        return n, 0
    elif r == 0:
        return n - 1, non_overlap
    else:
        return n, (x - non_overlap * n) % length


def generate_valid_lengths(valid_lengths, length, overlap):
    if valid_lengths.ndim > 0:
        res = []
        for vl in valid_lengths:
            n_segments, remainder = f(vl, length, overlap)
            full_segment_lengths = np.repeat(length, n_segments)
            if remainder == 0:
                arr = full_segment_lengths.astype(np.int32)
            else:
                arr = np.r_[full_segment_lengths, remainder].astype(np.int32)
            res.append(arr)
    else:
        n_segments, remainder = f(valid_lengths, length, overlap)
        full_segment_lengths = np.repeat(length, n_segments)
        if remainder == 0:
            res = full_segment_lengths.astype(np.int32)
        else:
            res = np.r_[full_segment_lengths, remainder].astype(np.int32)

    return res


class SegmentedArticlesDataset(Dataset):
    def __init__(self, dataset, bert_tokenizer, seg_len, shift,
                 pad, pair):
        self.transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=seg_len, pad=pad, pair=pair)

        articles = []
        labels = []

        for sample, label in dataset:
            # todo: treat cases where pad, pair are reverse
            tokens = bert_tokenizer(sample)
            segments = [
                space.sub(' ', ''.join(toks))
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
        # TODO: does this
        segments = []
        labels = []

        for article, label in articles_dataset:
            for segment in article:
                segments.append(segment)
                labels.append(label)

        return cls(segments, labels)

    def __getitem__(self, i):
        # TODO: make this compatible with base class
        return (self.segments[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))

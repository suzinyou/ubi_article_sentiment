import numpy as np
import torch


class BERTOutputSequence(object):
    def __init__(self, dataset, bert_clf, device, batch_size=64, random_seed=None):
        articles = []
        labels = []
        for sample, label in dataset:
            bert_outputs = []
            for token_ids, valid_length, segment_ids in sample:
                token_ids = torch.reshape(torch.Tensor(token_ids), (1, -1)).long().to('cpu')
                segment_ids = torch.reshape(torch.Tensor(segment_ids), (1, -1)).long().to('cpu')
                valid_length = valid_length.reshape(1,)

                # Get BERT output (batch_size, 768)
                # TODO: token ids, etc. must have ndim=2 !!!
                attention_mask = bert_clf.gen_attention_mask(token_ids, valid_length)
                _, pooler = bert_clf.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                                          attention_mask=attention_mask.float().to(token_ids.device))
                bert_outputs.append(pooler)
            bert_output_seq = torch.cat(bert_outputs)
            articles.append(bert_output_seq)
            labels.append(label)

        self.articles = np.arange(articles)
        self.labels = np.arange(labels)
        self.device = device
        self.size = len(dataset)
        self.batch_size = batch_size
        self._n_batches = self.size // self.batch_size + int(self.size % self.batch_size > 0)
        self.random_seed = 0 if random_seed is None else random_seed

    def __getitem__(self, item):
        return (self.articles[item], self.labels[item])

    def __iter__(self):
        np.random.seed(self.random_seed)
        shuffle_idx = np.arange(self.size)
        np.shuffle(shuffle_idx)

        _articles = self.articles[shuffle_idx]
        _labels = self.labels[shuffle_idx]

        for i in range(self._n_batches):
            # TODO: pad to max sequence length.
            articles = torch.nn.utils.rnn.pad_sequence(_articles[i:i+self.batch_size]).to(self.device)
            labels = torch.Tensor(_labels[i:i+self.batch_size]).to(self.device)

            yield articles, labels

    def __len__(self):
        return self._n_batches
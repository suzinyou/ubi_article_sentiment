import functools
import torch
from torch import nn

from article_sentiment.kobert.mixout import MixLinear


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


class BERTMixout(nn.Module):
    def __init__(self, bert, keep_prob=0.9):
        super(BERTMixout, self).__init__()
        self.bert = bert
        self.keep_prob = keep_prob

        setattr_tuple_list = []
        for name, module in bert.named_modules():
            if name.endswith('dropout') and isinstance(module, nn.Dropout):
                setattr_tuple_list.append((name, nn.Dropout(0)))
            elif (
                    name.endswith('attention.self.query') or
                    name.endswith('output.dense') or
                    name.endswith('intermediate.dense')
            ) and isinstance(module, nn.Linear):
                target_state_dict = module.state_dict()
                bias = True if module.bias is not None else False
                new_module = MixLinear(module.in_features, module.out_features,
                                       bias, target_state_dict['weight'], self.keep_prob)
                new_module.load_state_dict(target_state_dict)
                setattr_tuple_list.append((name, new_module))

        for name, module in setattr_tuple_list:
            rsetattr(self.bert, name, module)

    def forward(self, *input, **kwargs):
        self.bert.forward(*input, **kwargs)


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

        if dr_rate is not None and dr_rate != 0.0:
            self.dropout = nn.Dropout(p=dr_rate)

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))

        out = self.fc1(pooler)
        out = self.relu(out)
        if self.dr_rate:
            out = self.dropout(out)
        # else:
        #     out = pooler
        out = self.fc2(out)
        return out

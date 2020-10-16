from torch import nn


class RoBERT(nn.Module):
    def __init__(self,
                 input_size=768,
                 lstm_hidden_size=100,
                 fc_hidden_size=30,
                 num_classes=3,
                 dr_rate=None,
                 params=None):
        super(RoBERT, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,)
        self.fc = nn.Linear(lstm_hidden_size, fc_hidden_size)
        self.classifier = nn.Linear(fc_hidden_size, num_classes)
        self.dr_rate = dr_rate

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, seq):
        out, (h_n, c_n) = self.lstm(seq)
        last_out = out[-1]
        if self.dr_rate:
            last_out = self.dropout(last_out)
        fc_out = self.fc(last_out)
        if self.dr_rate:
            fc_out = self.dropout(fc_out)
        return self.classifier(fc_out)
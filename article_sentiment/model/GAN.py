from torch import nn


class Generator(nn.Module):
    def __init__(self, dim_latent_z, num_hidden_layers, dr_rate, num_classes, dim_hidden, n_gpu=0):
        super(Generator, self).__init__()
        # self.n_gpu = n_gpu # What is this for??
        modules = []
        dim_in = dim_latent_z
        for i in range(num_hidden_layers):
            modules.extend([
                nn.Linear(dim_in, dim_hidden),
                nn.LeakyReLU(True),
                nn.Dropout(dr_rate)])
            dim_in = dim_hidden
        self.main = nn.Sequential(*modules)

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, dim_bert_out, dim_hidden, num_hidden_layers, dr_rate, num_classes, n_gpu=0):
        super(Discriminator, self).__init__()
        # self.n_gpu = n_gpu # What is this for??
        modules = [nn.Dropout(dr_rate)]
        dim_in = dim_bert_out
        for i in range(num_hidden_layers):
            modules.extend([
                nn.Linear(dim_in, dim_hidden),
                nn.LeakyReLU(True),
                nn.Dropout(dr_rate)])
            dim_in = dim_hidden
        self.main = nn.Sequential(*modules)
        self.logit = nn.Linear(dim_in, num_classes + 1)
        self.prob = nn.Softmax(dim=1)

    def forward(self, input):
        last_hidden_feature = self.main(input)
        logit = self.logit(last_hidden_feature)
        prob = self.prob(logit)
        return last_hidden_feature, logit, prob
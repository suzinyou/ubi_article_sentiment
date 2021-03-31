import torch
from torch import nn


class LossDiscriminator(nn.Module):
    def __init__(self, num_classes, is_training=True, epsilon=1e-8):
        super(LossDiscriminator, self).__init__()
        self.softmax_clf_real = nn.Softmax(dim=1)
        self.log_softmax_clf_real = nn.LogSoftmax(dim=1)

        # self.softmax_unsup_real = nn.Softmax(dim=1)
        # self.log_softmax_unsup_real = nn.LogSoftmax(dim=1)

        self.is_training = is_training
        self.epsilon = epsilon

        self.num_classes = num_classes

    def forward(self, logits_d_sup, probs_d_sup, probs_g, labels, is_labeled_mask):
        """
        :returns: (loss_d, probabilities) where probabilities is the classification prob.
        """
        # loss_d_supervised
        logits_real = logits_d_sup[:, 1:]  # Assuming the first column is for the 'fake' class
        probabilities = self.softmax_clf_real(logits_real)
        log_probabilities = self.log_softmax_clf_real(logits_real)

        one_hot_labels = nn.functional.one_hot(labels, num_classes=self.num_classes)

        if self.is_training:
            per_example_loss = -torch.sum(one_hot_labels * log_probabilities, -1)
            per_example_loss = torch.masked_select(per_example_loss, is_labeled_mask)
            labeled_count = per_example_loss.size()
            loss_d_supervised = torch.div(
                torch.sum(per_example_loss),
                torch.max(torch.tensor(labeled_count), torch.tensor(1))
            )
        else:
            per_example_loss = -torch.sum(one_hot_labels * log_probabilities, -1)
            loss_d_supervised = torch.mean(per_example_loss)

        # loss_d_unsupervised_real
        loss_d_unsupervised_real = -torch.mean(torch.log(1 - probs_d_sup[:, 0] + self.epsilon))
        # Assuming the first column is for the 'fake' class

        # loss_d_unsupervised_fake
        loss_d_unsupervised_fake = -torch.mean(torch.log(probs_g[:, 0] + self.epsilon))

        loss_d = loss_d_supervised + loss_d_unsupervised_real + loss_d_unsupervised_fake
        return loss_d, probabilities


class LossGenerator(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(LossGenerator, self).__init__()
        self.epsilon = epsilon

    def forward(self, probs_g, features_g, features_d):
        # loss_g_fake
        loss_g_fake = -torch.mean(torch.log(1 - probs_g[:, 0] + self.epsilon))
        # Assuming the first column is for the 'fake' class

        # loss_g_feature_match
        loss_g_feature_match = torch.mean(torch.square(torch.mean(features_d, 0) - torch.mean(features_g, 0)))

        loss_g = loss_g_fake + loss_g_feature_match
        return loss_g
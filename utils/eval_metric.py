"""
This function is used to get the evaluation metrics for training quickly by applying confusion Matrix
The shape of the predict or target can be :
1.(N)
2.(N, C)
3.(N, d1, d2, ..., dk)
4.(N, C, d1, d2, ..., dk)
"""
import torch
import numpy as np

class ConfusionMeter():

    def __init__(self, num_class):

        self.num_class = num_class

        self.cf = np.ndarray((num_class, num_class), dtype=np.int32).reshape(num_class, num_class)
        self.reset()

        self.one_cf = np.ndarray((num_class, num_class), dtype=np.int32).reshape(num_class, num_class)
        self.one_reset()

    def reset(self):
        self.cf.fill(0)

    def one_reset(self):
        self.one_cf.fill(0)

    def process(self, predicted, target):
        if torch.is_tensor( predicted ):

            predicted = predicted.cpu().numpy()
            target = target.cpu().numpy()
            batch_size = 1
        else:
            batch_size = predicted.shape[0]

        predicted = predicted.reshape(batch_size, -1)
        target = target.reshape(batch_size, -1)

        return predicted.astype(np.int32), target.astype(np.int32)

    def update(self, predicted, target):

        predicted, target = self.process(predicted, target)

        for p, t in zip(predicted, target):
            position = t * self.num_class + p

            self.one_cf = np.bincount(position, minlength=self.num_class ** 2). \
                reshape(self.num_class, self.num_class)

            self.cf += self.one_cf

    # here to calculate the training metrics: accuracy, Iou, dice, recall and precision
    def get_scores(self, metrics=None, is_single=False):

        if is_single:
            cf = self.one_cf
        else:
            cf = self.cf

        epsilon = 1e-20

        acc = np.diag(cf).sum() / (cf.sum() + epsilon)

        IoU = np.diag(cf) / ((np.sum(cf, axis=0) + np.sum(cf, axis=1) - np.diag(cf)) + epsilon)
        mean_IoU = np.nanmean(IoU)

        dice = 2 * np.diag(cf) / ((np.sum(cf, axis=0) + np.sum(cf, axis=1)) + epsilon)
        mean_dice = dice[1]

        recall = np.diag(cf) / (np.sum(cf, axis=1) + epsilon)
        mean_recall = recall[1]

        precision = np.diag(cf) / (np.sum(cf, axis=0) + epsilon)
        precision_positive = precision[1]

        scores = {
            'IoU': mean_IoU,
            'Dice': mean_dice,
            'Acc': acc,
            'Recall': mean_recall,
            'Precision': precision_positive
        }

        if metrics == None:
            return scores

        else:
            return scores[metrics]




import torch
import random


class SampleIndexGenerater():
    """ Sample Index Generator for ML-AE baseline """

    def __init__(self, data, batch_size, choos_center, method='near'):

        self.num_train_sample = data.shape[0]
        self.batch_size = batch_size
        self.method = method
        self.choos_center = choos_center
        self.Reset()

    def CalPairwiseDistance(self, data):
        # use the torch to calculate the PairwiseDistance
        x = data.float()
        y = data.float()
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        d = dist.clamp(min=1e-12).sqrt()  # for numerical stabili
        return d

    def Reset(self):
        # used to reset the unuse_index every epoch
        self.unuse_index = torch.randperm(self.num_train_sample).tolist()

    def CalSampleIndex(self, batch_idx):

        def PutInUseindex(use_index, unuse_index, i):
            if i in unuse_index:
                use_index.append(i)
                unuse_index.remove(i)
                return True
            else:
                return False

        if self.method == 'normal':
            # Prepare data and label
            use_index = self.unuse_index[:self.batch_size]
            self.unuse_index = self.unuse_index[self.batch_size:]

        elif self.method == 'near':
            # calculate the near point for every random point
            self.choos_near = self.batch_size // (self.choos_center) - 1
            # init the list of use index
            use_index = []

            # put the random point into the use_index list
            # and remove it from the unuse_index
            for i in range(self.choos_center):
                PutInUseindex(use_index, self.unuse_index, self.unuse_index[0])

            # put the near point for every random point into the
            # use_index list and remove it from the unuse_index
            for i in range(self.choos_center):
                random_point = use_index[i]
                for j in range(self.choos_near):
                    cont = 1
                    while cont < self.num_train_sample and \
                            (
                                not PutInUseindex(
                                    use_index,
                                    self.unuse_index,
                                    self.nearindex[random_point][cont]
                                )
                            ):
                        cont += 1

            # shuffle the points for learning better
            random.shuffle(use_index)

        return use_index

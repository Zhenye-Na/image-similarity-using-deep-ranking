"""
Image Similarity using Deep Ranking

references: https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42945.pdf

@author: Zhenye Na
"""

from torch.utils.data.sampler import Sampler

class TripletSampler(Sampler):
    """
    Samples elements more than once in a single pass through the data.

    This allows the number of samples per epoch to be larger than the number
    of samples itself, which can be useful for data augmentation.
    """
    def __init__(self, nb_samples, desired_samples, shuffle=False):
        self.data_samples = nb_samples
        self.desired_samples = desired_samples
        self.shuffle=shuffle

    def _gen_sample_array(self):
        n_repeats = self.desired_samples / self.data_samples
        self.sample_idx_array = torch.range(0,self.data_samples-1).repeat(n_repeats).long()
        # if self.shuffle:
        #   self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array)]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self._gen_sample_array())

    def __len__(self):
        return self.desired_samples

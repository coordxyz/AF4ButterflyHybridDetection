import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np
import pdb

def CreateDataset(opt):
    from data.aligned_dataset import BootAF_I2DE_trainDataset
    dataset = BootAF_I2DE_trainDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateDataset_test(opt):
    from data.aligned_dataset import BootAF_I2DE_testDataset    
    dataset = BootAF_I2DE_testDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        if opt.isTrain:
            self.dataset = CreateDataset(opt)
        else:
            self.dataset = CreateDataset_test(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=0,
            worker_init_fn=lambda _: np.random.seed())

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
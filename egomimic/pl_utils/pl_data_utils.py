from torch.utils.data import DataLoader, random_split
from lightning import LightningDataModule
from egomimic.utils.egomimicUtils import nds
import json
import os
from rldb.utils import RLDBDataset
from termcolor import cprint

class RLDBModule(LightningDataModule):
    def __init__(self, dataset, dataloader_kwargs):
        super().__init__()
        self.dataset1 = dataset
        self.dataloader_kwargs = dataloader_kwargs

        # self.train_dataset, self.val_dataset = random_split(self.dataset1, [0.8, 0.2])
        self.train_dataset, self.val_dataset = self.dataset1, self.dataset1
        cprint(f"Warning: RLDBModule is using the same dataset for train and val", "yellow")

    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, shuffle=False, **self.dataloader_kwargs)


class DualDataModuleWrapper(LightningDataModule):
    """
    Same as DataModuleWrapper but there are two train datasets and two valid datasets
    """

    def __init__(
        self,
        train_dataset1,
        valid_dataset1,
        train_dataset2,
        valid_dataset2,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset1 = train_dataset1
        self.valid_dataset1 = valid_dataset1
        self.train_dataset2 = train_dataset2
        self.valid_dataset2 = valid_dataset2
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader1 = DataLoader(
            dataset=self.train_dataset1, **self.train_dataloader_params
        )
        new_dataloader2 = DataLoader(
            dataset=self.train_dataset2, **self.train_dataloader_params
        )
        return [new_dataloader1, new_dataloader2]

    def val_dataloader_1(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset1, **self.valid_dataloader_params
        )
        return new_dataloader

    def val_dataloader_2(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset2, **self.valid_dataloader_params
        )
        return new_dataloader

    # def val_dataloader(self):
    #     new_dataloader1 = DataLoader(dataset=self.valid_dataset1, **self.valid_dataloader_params)
    #     new_dataloader2 = DataLoader(dataset=self.valid_dataset2, **self.valid_dataloader_params)
    #     return [new_dataloader1, new_dataloader2]


class DataModuleWrapper(LightningDataModule):
    """
    Wrapper around a LightningDataModule that allows for the data loader to be refreshed
    constantly.
    """

    def __init__(
        self,
        train_dataset,
        valid_dataset,
        train_dataloader_params,
        valid_dataloader_params,
    ):
        """
        Args:
            data_module_fn (function): function that returns a LightningDataModule
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_dataloader_params = train_dataloader_params
        self.valid_dataloader_params = valid_dataloader_params

    def train_dataloader(self):
        new_dataloader = DataLoader(
            dataset=self.train_dataset, **self.train_dataloader_params
        )
        return new_dataloader

    def val_dataloader_1(self):
        new_dataloader = DataLoader(
            dataset=self.valid_dataset, **self.valid_dataloader_params
        )
        return new_dataloader


def get_dual_data_module(
    trainset, trainset_2, validset, validset_2, train_sampler, valid_sampler, config
):
    return DualDataModuleWrapper(
        train_dataset1=trainset,
        valid_dataset1=validset,
        train_dataset2=trainset_2,
        valid_dataset2=validset_2,
        train_dataloader_params=dict(
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
            pin_memory=True,
        ),
        valid_dataloader_params=dict(
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_data_workers,
            drop_last=True,
            pin_memory=True,
        ),
    )


def get_data_module(trainset, validset, train_sampler, valid_sampler, config):
    return DataModuleWrapper(
        train_dataset=trainset,
        valid_dataset=validset,
        train_dataloader_params=dict(
            sampler=train_sampler,
            batch_size=config.train.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.train.num_data_workers,
            drop_last=True,
            pin_memory=True,
        ),
        valid_dataloader_params=dict(
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.num_data_workers,
            drop_last=True,
            pin_memory=True,
        ),
    )

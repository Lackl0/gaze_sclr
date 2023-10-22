import torch.utils.data
from typing import Tuple, Union

import yacs.config
from torch.utils.data import DataLoader


class InfiniteBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        while True:
            batch = []
            for idx in self.sampler:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch


class FastDataloader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size, num_workers, drop_last, 
                 collate_fn=None, pin_memory=True, sampler=None):
        if sampler is None:
            sampler = torch.utils.data.sampler.RandomSampler(dataset)
        self.sampler_origin = sampler  # to access DistributedSampler later
        batch_sampler = InfiniteBatchSampler(sampler, batch_size, drop_last)

        super().__init__(dataset=dataset, 
                         batch_sampler=batch_sampler, 
                         num_workers=num_workers,
                         collate_fn=collate_fn, 
                         pin_memory=pin_memory)

        self.data_iter = super().__iter__()
        self.data_idx = 0

    def __iter__(self):
        while self.data_idx < len(self.data_iter):
            self.data_idx += 1
            yield next(self.data_iter)

        self.data_idx = 0

def create_dataloader(
        config: yacs.config.CfgNode,
        is_train: bool) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    if is_train:
        train_dataset, val_dataset = create_dataset(config, is_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.train_dataloader.num_workers,
            pin_memory=config.train.train_dataloader.pin_memory,
            drop_last=config.train.train_dataloader.drop_last,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=config.train.val_dataloader.num_workers,
            pin_memory=config.train.val_dataloader.pin_memory,
            drop_last=False,
        )
        return train_loader, val_loader
    else:
        test_dataset = create_dataset(config, is_train)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.test.batch_size,
            num_workers=config.test.dataloader.num_workers,
            shuffle=False,
            pin_memory=config.test.dataloader.pin_memory,
            drop_last=False,
        )
        return test_loader

def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool = True) -> Union[List[Dataset], Dataset]:
    if config.mode == GazeEstimationMethod.MPIIGaze.name:
        from .mpiigaze import OnePersonDataset
    elif config.mode == GazeEstimationMethod.MPIIFaceGaze.name:
        from .mpiifacegaze import OnePersonDataset
    else:
        raise ValueError

    dataset_dir = pathlib.Path(config.dataset.dataset_dir)

    assert dataset_dir.exists()
    assert config.train.test_id in range(-1, 15)
    assert config.test.test_id in range(15)
    person_ids = [f'p{index:02}' for index in range(15)]

    transform = create_transform(config)

    if is_train:
        if config.train.test_id == -1:
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids
            ])
            assert len(train_dataset) == 45000
        else:
            test_person_id = person_ids[config.train.test_id]
            train_dataset = torch.utils.data.ConcatDataset([
                OnePersonDataset(person_id, dataset_dir, transform)
                for person_id in person_ids if person_id != test_person_id
            ])
            assert len(train_dataset) == 42000

        val_ratio = config.train.val_ratio
        assert val_ratio < 1
        val_num = int(len(train_dataset) * val_ratio)
        train_num = len(train_dataset) - val_num
        lengths = [train_num, val_num]
        return torch.utils.data.dataset.random_split(train_dataset, lengths)
    else:
        test_person_id = person_ids[config.test.test_id]
        test_dataset = OnePersonDataset(test_person_id, dataset_dir, transform)
        assert len(test_dataset) == 3000
        return test_dataset
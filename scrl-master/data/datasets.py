import logging
import random
import h5py
from torch.utils.data import Dataset

import torch
from torchvision import datasets

log = logging.getLogger('main')


def get_dataset(data_name: str, data_root: str, train: bool, 
                transform, num_subsample=-1):
    # dataset
    if data_name == 'imagenet':
        dataset_cls = get_dataset_cls(data_name)
        split = 'train' if train else 'val'
        dataset = dataset_cls(root=data_root, split=split, transform=transform)
        num_classes = len(dataset.classes)
    elif data_name == 'mpiifacegaze':
        dataset = OnePersonDataset(person_id_str="p00", dataset_path=data_root, transform=transform)
        num_classes = 2  # 视线估计的类别数，根据实际情况调整
    else:
        raise ValueError(f"Unknown dataset name: {data_name}")

    log.info(f"[Dataset] {data_name}(train={'True' if train else 'False'}) / "
             f"{dataset.__len__()} images are available.")

    # use smaller subset when debugging
    if num_subsample > 0:
        log.info(f"[Dataset] sample {num_subsample} images from the dataset in "
                 f"{'train' if train else 'test'} set.")
        num_subsample = min(num_subsample, len(dataset))
        indices = random.choices(range(len(dataset)), k=num_subsample)
        dataset = torch.utils.data.Subset(dataset, list(indices))
        
    return dataset, num_classes


def get_dataset_cls(name):
    try:
        return {
            'imagenet': datasets.ImageNet,
            # add your custom dataset class here
        }[name]
    except KeyError:
        raise KeyError(f"Unexpected dataset name: {name}")


class OnePersonDataset(Dataset):
    def __init__(self, person_id_str: str, dataset_path: pathlib.Path,
                 transform: Callable):
        self.person_id_str = person_id_str
        self.dataset_path = pathlib.Path(dataset_path)
        self.transform = transform

    def __getitem__(
            self,
            index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.dataset_path, 'r') as f:
            image = f.get(f'{self.person_id_str}/image/{index:04}')[()]
            pose = f.get(f'{self.person_id_str}/pose/{index:04}')[()]
            gaze = f.get(f'{self.person_id_str}/gaze/{index:04}')[()]
        image = self.transform(image)
        pose = torch.from_numpy(pose)
        gaze = torch.from_numpy(gaze)
        return image, pose, gaze

    def __len__(self) -> int:
        return 3000
import os

from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Callable, Optional

DATA_TRANSFORM = {
    'train': transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ]
    ),
    'val': transforms.Compose(
        [
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
        ]
    )
}


class MnistDataset(Dataset):

    def __init__(self, image_dir: str, transformer: Optional[Callable] = None):
        image_suffix = ['.png', '.jpg']
        self.transformer = transformer
        self.images_path = [f for f in Path(image_dir).iterdir() if f.suffix in image_suffix]

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = Image.open(image_path)
        image_name = os.path.basename(image_path)
        image_label = image_name.split('_')[0]
        if self.transformer:
            image = self.transformer(image)

        return image, int(image_label)

    def __len__(self):
        return len(self.images_path)

import logging

import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, noise_rate=0):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.noise_rate = noise_rate
        self.data, self.target = self.__build_truncated_dataset__()


    def corrupt_label(self, y_train, noise_rate):
        """Corrupts training labels.

        Args:
        y_train: training labels
        noise_rate: input noise ratio

        Returns:
        corrupted_y_train: corrupted training labels
        noise_idx: corrupted index
        """

        if(noise_rate == 0.0):
            return y_train, []
        y_set = list(set(y_train))

        # Sets noise_idx
        temp_idx = np.random.permutation(len(y_train))
        noise_idx = temp_idx[:int(len(y_train) * noise_rate)]

        # Corrupts label
        corrupted_y_train = y_train[:]

        for itt in noise_idx:
            temp_y_set = y_set[:]
            del temp_y_set[y_train[itt]]
            rand_idx = np.random.randint(len(y_set) - 1)
            corrupted_y_train[itt] = temp_y_set[rand_idx]

        return corrupted_y_train, noise_idx

    def __build_truncated_dataset__(self):
        print("download = " + str(self.download))
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            # print("train member of the class: {}".format(self.train))
            # data = cifar_dataobj.train_data
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
            if(self.noise_rate != 0):
                target, noise_idx = self.corrupt_label(target, self.noise_rate)
                target = np.delete(target, noise_idx, axis=0)
                data = np.delete(data, noise_idx, axis=0)

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

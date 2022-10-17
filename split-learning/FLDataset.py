import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np

class CustomDataset(Dataset):

    def __init__(self, path):
        # Initialize data, download, etc.
        # read with numpy or pandas
        super(CustomDataset, self).__init__()
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32, skiprows=1, usecols=range(2,20))
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class CustomCIFARDataset(datasets.CIFAR10):

    def __init__(self, root, train, download, transform, client_num):
        super().__init__(root='./data', train=True, download=True, transform=transform)
        self.client_num = client_num    

    def __getitem__(self, index):

        data, target = super().__getitem__(index)
        height = data[0].shape[-1]//10
        data = data[:, :, height * self.client_num : height * (self.client_num + 1)]
        return data, target 


    

def makeIID(dataset, num_users):
    points = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, points, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict
    
    
def mnistIID(dataset, num_users):
    images = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, images, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict

def mnistNonIID(dataset, num_users):
    classes, images = 200, 300
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    for i in range(num_users):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
    return users_dict

def mnistNonIIDUnequal(dataset, num_users):
    classes, images = 1200, 50
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    min_cls_per_client = 1
    max_cls_per_client = 30

    random_selected_classes = np.random.randint(min_cls_per_client, max_cls_per_client+1, size=num_users)
    random_selected_classes = np.around(random_selected_classes / sum(random_selected_classes) * classes)
    random_selected_classes = random_selected_classes.astype(int)

    if sum(random_selected_classes) > classes:
        for i in range(num_users):
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, 1, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)

        random_selected_classes = random_selected_classes-1

        for i in range(num_users):
            if len(classes_indx) == 0:
                continue
            class_size = random_selected_classes[i]
            if class_size > len(classes_indx):
                class_size = len(classes_indx)
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
    else:

        for i in range(num_users):
            class_size = random_selected_classes[i]
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)

        if len(classes_indx) > 0:
            class_size = len(classes_indx)
            j = min(users_dict, key=lambda x: len(users_dict.get(x)))
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[j] = np.concatenate((users_dict[j], indeces[t*images:(t+1)*images]), axis=0)

    return users_dict


def cifarIID(dataset, num_users):
    images = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, images, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict



def load_dataset(num_users, iidtype, dataset_name):
    tranform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    # Load MNIST dataset     
    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=tranform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=tranform)
        train_group, test_group = None, None
        if iidtype == 'iid':
            train_group = mnistIID(train_dataset, num_users)
            test_group = mnistIID(test_dataset, num_users)
        elif iidtype == 'noniid':
            train_group = mnistNonIID(train_dataset, num_users)
            test_group = mnistNonIID(test_dataset, num_users)
        else:
            train_group = mnistNonIIDUnequal(train_dataset, num_users)
            test_group = mnistNonIIDUnequal(test_dataset, num_users)
        return train_dataset, test_dataset, train_group, test_group
    ########################
    
    if dataset_name == "CIFAR-10":
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
        train_group, test_group = None, None
        if iidtype == 'iid':
            train_group = cifarIID(train_dataset, num_users)
            test_group = cifarIID(test_dataset, num_users)
        return train_dataset, test_dataset, train_group, test_group
    
    
class FedDataset(Dataset):
    def __init__(self, dataset, indx):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]
        
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
        images, label = self.dataset[self.indx[item]]
        return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()

def getActualImgs(dataset, indeces, batch_size):
    return DataLoader(FedDataset(dataset, indeces), batch_size=128, shuffle=True, num_workers=2)

def getActualDataPoints(dataset, indeces, batch_size):
    return CustomDataset(FedDataset(dataset, indeces), batch_size=batch_size, shuffle=True)



def dataset_meta(dataset_name = "MNIST"):
    if dataset_name == "MINIST":
        return 6000
    elif dataset_name == "CIFAR-10":
        return 6000
    
def load_global_test_dataset(dataset_name = "MNIST"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    if dataset_name == "MNIST":
        return datasets.MNIST('./data', train=False, download=True, transform=transform)
    elif dataset_name == "CIFAR-10":
        return datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
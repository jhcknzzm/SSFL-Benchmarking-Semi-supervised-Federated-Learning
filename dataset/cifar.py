import logging

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
import copy
from .randaugment import RandAugmentMC
import random
import numpy as np

seed_value = 1

random.seed(seed_value)
np.random.seed(seed_value)

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)


def get_cifar10(root, num_expand_x, num_expand_u,device_ids, server_idxs):
    root='./data'
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=False)

    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    #     base_dataset.targets, num_expand_x, num_expand_u, device_ids,server_idxs)
    train_labeled_idxs, train_unlabeled_idxs = server_idxs, device_ids

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset_list = []
    for id in range(len(train_unlabeled_idxs)):
        train_unlabeled_dataset = CIFAR10SSL(
            root, train_unlabeled_idxs[id], train=True,
            transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

        train_unlabeled_dataset_list.append(train_unlabeled_dataset)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)
    logger.info("Dataset: CIFAR10")

    return train_labeled_dataset, train_unlabeled_dataset_list, test_dataset, base_dataset


def get_cifar10_semi(root, num_expand_x, num_expand_u,device_ids, server_idxs):
    root='./data'
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=False)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split_semi_cifar(
        base_dataset.targets, num_expand_x, num_expand_u, device_ids, server_idxs)
    # train_labeled_idxs, train_unlabeled_idxs = server_idxs, device_ids


    train_unlabeled_dataset_list = []
    train_labeled_dataset_list = []
    for id in range(len(train_unlabeled_idxs)):
        print(id)
        train_unlabeled_dataset = CIFAR10SSL(
            root, train_unlabeled_idxs[id], train=True,
            transform=TransformFix(mean=cifar10_mean, std=cifar10_std))

        train_labeled_dataset = CIFAR10SSL(
            root, train_labeled_idxs[id], train=True,
            transform=transform_labeled)

        train_unlabeled_dataset_list.append(train_unlabeled_dataset)
        train_labeled_dataset_list.append(train_labeled_dataset)

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)
    logger.info("Dataset: CIFAR10")


    return train_labeled_dataset_list, train_unlabeled_dataset_list, test_dataset, base_dataset

def get_svhn(root, num_expand_x, num_expand_u,device_ids, server_idxs):
    root='./data'

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.SVHN(root, split='train', download=False)

    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(
    #     base_dataset.labels, num_expand_x, num_expand_u, device_ids,server_idxs)
    train_labeled_idxs, train_unlabeled_idxs = server_idxs, device_ids

    train_labeled_dataset = SVHNSSL(
        root, train_labeled_idxs, split='train',
        transform=transform_labeled)

    train_unlabeled_dataset_list = []
    train_unlabeled_idxs_tmp = copy.deepcopy(train_unlabeled_idxs[0])

    import functools
    import operator

    for id in range(len(train_unlabeled_idxs)):
        train_unlabeled_dataset = SVHNSSL(
            root, train_unlabeled_idxs[id], split='train',
            transform=TransformFix(mean=cifar10_mean, std=cifar10_std))
        train_unlabeled_dataset_list.append(train_unlabeled_dataset)

    test_dataset = datasets.SVHN(
        root, split='train', transform=transform_val, download=False)
    logger.info("Dataset: SVHN")


    return train_labeled_dataset, train_unlabeled_dataset_list, test_dataset, base_dataset

def get_cifar100(root, num_labeled, num_expand_x, num_expand_u):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        base_dataset.targets, num_classes=100)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFix(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    logger.info("Dataset: CIFAR100")
    logger.info(f"Labeled examples: {len(train_labeled_idxs)}"
                f" Unlabeled examples: {len(train_unlabeled_idxs)}")

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset

def get_emnist(root, num_expand_x, num_expand_u,device_ids, server_idxs, attack_idxs=None):
    root='./data'
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    base_dataset = datasets.EMNIST(root, train=True,split='balanced', download=True)


    # train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        # base_dataset.targets, num_expand_x, num_expand_u, device_ids,server_idxs)
    train_labeled_idxs, train_unlabeled_idxs = server_idxs, device_ids

    train_labeled_dataset = EMNIST(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    if attack_idxs is not None:
        train_attack_dataset = EMNIST(
            root, attack_idxs, train=True,
            transform=transform_labeled)


    train_unlabeled_dataset_list = []
    print('len(train_unlabeled_idxs):',len(train_unlabeled_idxs))


    for id in range(len(train_unlabeled_idxs)):
        train_unlabeled_dataset = EMNIST(
            root, train_unlabeled_idxs[id], train=True,
            transform=TransformFix(size = 28, mean=(0.1307,), std=(0.3081,)))
        train_unlabeled_dataset_list.append(train_unlabeled_dataset)

    test_dataset = datasets.EMNIST(
        root, train=False,split='balanced', transform=transform_val, download=True)

    if attack_idxs is not None:
        return train_labeled_dataset, train_unlabeled_dataset_list, test_dataset, train_attack_dataset, base_dataset
    else:
        return train_labeled_dataset, train_unlabeled_dataset_list, test_dataset, base_dataset

def get_emnist_semi(root, num_expand_x, num_expand_u,device_ids, server_idxs):
    root='./data'
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    base_dataset = datasets.EMNIST(root, train=True,split='balanced', download=True)


    train_labeled_idxs, train_unlabeled_idxs = x_u_split_semi(
        base_dataset.targets, num_expand_x, num_expand_u, device_ids, server_idxs)


    train_unlabeled_dataset_list = []
    train_labeled_dataset_list = []
    train_unlabeled_idxs_tmp = copy.deepcopy(train_unlabeled_idxs[0])



    for id in range(len(train_unlabeled_idxs)):
        train_unlabeled_dataset = EMNIST(
            root, train_unlabeled_idxs[id], train=True,
            transform=TransformFix(size = 28, mean=(0.1307,), std=(0.3081,)))
        train_unlabeled_dataset_list.append(train_unlabeled_dataset)

        train_labeled_dataset = EMNIST(
            root, train_labeled_idxs[id], train=True,
            transform=transform_labeled)
        train_labeled_dataset_list.append(train_labeled_dataset)

    test_dataset = datasets.EMNIST(
        root, train=False,split='balanced', transform=transform_val, download=True)


    return train_labeled_dataset_list, train_unlabeled_dataset_list, test_dataset

def x_u_split(labels,
              num_expand_x,
              num_expand_u,
              device_ids,
              server_idxs):
    labels = np.array(labels)
    labeled_idx = copy.deepcopy(server_idxs)
    unlabeled_idx = []

    unlabeled_idx_list = []
    for id in range(len(device_ids)):
        unlabeled_idx = device_ids[id]

        exapand_unlabeled = num_expand_u // len(device_ids[id]) // len(device_ids)

        unlabeled_idx = np.hstack(
            [unlabeled_idx for _ in range(exapand_unlabeled)])

        if len(unlabeled_idx) < num_expand_u // len(device_ids):
            diff = num_expand_u // len(device_ids) - len(unlabeled_idx)
            unlabeled_idx = np.hstack(
                (unlabeled_idx, np.random.choice(unlabeled_idx, diff)))
        else:
            assert len(unlabeled_idx) == num_expand_u // len(device_ids)

        unlabeled_idx_list.append(unlabeled_idx)


    exapand_labeled = num_expand_x // len(labeled_idx)
    labeled_idx = np.hstack(
        [labeled_idx for _ in range(exapand_labeled)])
    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    return labeled_idx, unlabeled_idx_list

def x_u_split_semi(labels,
              num_expand_x,
              num_expand_u,
              device_ids,
              server_idxs):


    server_semi_idxs = []
    for i in range(len(device_ids)):
        server_semi_idxs.append([])

    num = len(server_idxs)//len(device_ids)
    for id in range(len(device_ids)-1):
        idx_tmp = server_idxs[id*num:(id+1)*num]
        server_semi_idxs[id] = idx_tmp
    server_semi_idxs[len(device_ids)-1] = server_idxs[(id+1)*num:]
    labels = np.array(labels)
    labeled_idx = copy.deepcopy(server_idxs)
    unlabeled_idx = []

    unlabeled_idx_list = []
    for id in range(len(device_ids)):
        unlabeled_idx = device_ids[id]
        exapand_unlabeled = num_expand_u // len(device_ids[id]) // len(device_ids)

        unlabeled_idx = np.hstack(
            [unlabeled_idx for _ in range(exapand_unlabeled)])

        if len(unlabeled_idx) < num_expand_u // len(device_ids):
            diff = num_expand_u // len(device_ids) - len(unlabeled_idx)
            unlabeled_idx = np.hstack(
                (unlabeled_idx, np.random.choice(unlabeled_idx, diff)))
        else:
            assert len(unlabeled_idx) == num_expand_u // len(device_ids)

        unlabeled_idx_list.append(unlabeled_idx)

    labeled_idx_list = []
    for id in range(len(device_ids)):
        labeled_idx = server_semi_idxs[id]
        exapand_unlabeled = num_expand_u // len(server_semi_idxs[id]) // len(server_semi_idxs)

        labeled_idx = np.hstack(
            [labeled_idx for _ in range(exapand_unlabeled)])

        if len(labeled_idx) < num_expand_u // len(device_ids):
            diff = num_expand_u // len(device_ids) - len(labeled_idx)
            labeled_idx = np.hstack(
                (labeled_idx, np.random.choice(labeled_idx, diff)))
        else:
            assert len(labeled_idx) == num_expand_u // len(device_ids)

        labeled_idx_list.append(labeled_idx)

    return labeled_idx_list, unlabeled_idx_list


def x_u_split_semi_cifar(labels,
              num_expand_x,
              num_expand_u,
              device_ids,
              server_idxs):

    unlabeled_idx = []
    unlabeled_idx_list = []
    for id in range(len(device_ids)):
        unlabeled_idx = device_ids[id]
        exapand_unlabeled = num_expand_u // len(device_ids[id]) // len(device_ids)

        unlabeled_idx = np.hstack(
            [unlabeled_idx for _ in range(exapand_unlabeled)])

        if len(unlabeled_idx) < num_expand_u // len(device_ids):
            diff = num_expand_u // len(device_ids) - len(unlabeled_idx)
            unlabeled_idx = np.hstack(
                (unlabeled_idx, np.random.choice(unlabeled_idx, diff)))
        else:
            assert len(unlabeled_idx) == num_expand_u // len(device_ids)

        unlabeled_idx_list.append(unlabeled_idx)

    labeled_idx_list = []
    for id in range(len(device_ids)):
        labeled_idx = server_idxs[id]

        exapand_unlabeled = num_expand_u // len(device_ids[id]) // len(device_ids)

        labeled_idx = np.hstack(
            [labeled_idx for _ in range(exapand_unlabeled)])

        if len(labeled_idx) < num_expand_u // len(device_ids):
            diff = num_expand_u // len(device_ids) - len(labeled_idx)
            labeled_idx = np.hstack(
                (labeled_idx, np.random.choice(labeled_idx, diff)))
        else:
            assert len(labeled_idx) == num_expand_u // len(device_ids)

        labeled_idx_list.append(labeled_idx)

    return labeled_idx_list, unlabeled_idx_list


class TransformFix(object):
    def __init__(self, mean, std,size=32):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class EMNIST(datasets.EMNIST):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=True,split='balanced'):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,split='balanced',
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.cpu().numpy()
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = target.cpu().numpy()
            target = self.target_transform(target)
        return img, target

class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split='train',
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

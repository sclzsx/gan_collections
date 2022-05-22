from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from uc_dataset import UCDataset, group1, group2, group3, all_names


def dataloader(dataset, input_size, batch_size, split='train', train_aug=False, use_crop=False, datadir=''):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        data_loader = DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST('data/fashion-mnist', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        print(len(datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform)))
        data_loader = DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN('data/svhn', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10('data/stl10', split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN('data/lsun', classes=['church_outdoor_train'], transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'uc':
        dataset = UCDataset(image_dir=datadir, train_aug=train_aug, choose_classes=all_names,
                            img_size=input_size, use_crop=use_crop)
        print(len(dataset))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # elif dataset == 'uc1':
    #     dataset = UCDataset('../UCMerced_LandUse/train64_offline_aug', train_aug=0, choose_classes=group1,
    #                         img_size=input_size)
    #     print(len(dataset))
    #     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # elif dataset == 'uc2':
    #     dataset = UCDataset('../UCMerced_LandUse/train64_offline_aug', train_aug=0, choose_classes=group2,
    #                         img_size=input_size)
    #     print(len(dataset))
    #     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # elif dataset == 'uc3':
    #     dataset = UCDataset('../UCMerced_LandUse/train64_offline_aug', train_aug=0, choose_classes=group3,
    #                         img_size=input_size)
    #     print(len(dataset))
    #     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return data_loader

import os
import cv2
from PIL import Image
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import torchvision.transforms.functional as f

# class_names = [i.name for i in Path('../UCMerced_LandUse/Images').iterdir() if i.is_dir()]
# class_ids = [i for i in range(len(class_names))]
# class_name2id = dict(zip(class_names, class_ids))
# class_id2name = dict(zip(class_ids, class_names))

group1 = 'agricultural,airplane,baseballdiamond,beach,buildings,chaparral,denseresidential'
group2 = 'forest,freeway,golfcourse,harbor,intersection,mediumresidential,mobilehomepark'
group3 = 'overpass,parkinglot,river,runway,sparseresidential,storagetanks,tenniscourt'
all_names = group1 + ',' + group2 + ',' + group3


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def split_ucdataset_for_wgangp_and_classify(dataset_dir, train_num=80, image_size=64):
    paths = [i for i in Path(dataset_dir).rglob('*.*')]

    train_dir = str(Path(dataset_dir).parent) + '/train' + str(image_size)
    test_dir = str(Path(dataset_dir).parent) + '/test' + str(image_size)
    mkdir(train_dir)
    mkdir(test_dir)

    ids = [i for i in range(100)]
    random.shuffle(ids)
    train_ids = ids[:train_num]

    for path in paths:
        image = Image.open(str(path)).convert('RGB')
        image = image.resize((image_size, image_size), Image.ANTIALIAS)
        id = int(path.name[-6:-4])
        if id in train_ids:
            image.save(train_dir + '/' + path.name[:-3] + 'jpg', quality=100)
        else:
            image.save(test_dir + '/' + path.name[:-3] + 'jpg', quality=100)

    print('done')


def offline_aug_data(data_dir, aug_num_each=100):
    save_dir = data_dir + '_offline_aug'
    mkdir(save_dir)
    for path in tqdm(Path(data_dir).glob('*.*')):
        im = Image.open(str(path))
        im.save(save_dir + '/' + path.name[:-4] + '_0.jpg', quality=100)

        for i in range(1, aug_num_each + 1):
            im1 = transforms.RandomHorizontalFlip(p=random.random())(im)  # p表示概率
            im2 = transforms.RandomVerticalFlip(p=random.random())(im1)
            # 分别代表亮度,对比度,饱和度,其中hue最大不超过0.5
            im3 = transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)(im2)
            # im3 = transforms.ColorJitter(hue=0.5)(im2)
            new_im = transforms.RandomRotation(random.randint(0, 5))(im3)  # 随机旋转0-5度

            new_im.save(save_dir + '/' + path.name[:-4] + '_' + str(i) + '.jpg', quality=100)


class UCDataset(Dataset):
    def __init__(self, image_dir, train_aug=0, choose_classes='', img_size=64, use_crop=0):
        if len(choose_classes) < 1:
            self.paths = [i for i in Path(image_dir).rglob('*.*')]
            class_names = [i.name for i in Path('../UCMerced_LandUse/Images').iterdir() if i.is_dir()]
            class_ids = [i for i in range(len(class_names))]
            self.class_name2id = dict(zip(class_names, class_ids))
        else:
            self.paths = []
            for i in Path(image_dir).rglob('*.*'):
                class_name = (i.name.split('.')[0]).split('_')[0][:-2]
                if class_name in choose_classes:
                    self.paths.append(i)
            class_names = choose_classes.split(',')
            class_ids = [i for i in range(len(class_names))]
            self.class_name2id = dict(zip(class_names, class_ids))
            # print(self.class_name2id)

        if use_crop:
            if train_aug:
                self.transform = transforms.Compose([
                    # transforms.RandomCrop(img_size, padding=4),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomCrop(img_size),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.RandomResizedCrop(img_size),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.RandomCrop(img_size),
                    transforms.RandomResizedCrop(img_size),
                    # transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        else:
            if train_aug:
                self.transform = transforms.Compose([
                    # transforms.RandomCrop(img_size, padding=4),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomCrop(img_size),
                    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    transforms.Resize((img_size, img_size), interpolation=f._interpolation_modes_from_int(0)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                self.transform = transforms.Compose([
                    # transforms.RandomCrop(img_size),
                    transforms.Resize((img_size, img_size), interpolation=f._interpolation_modes_from_int(0)),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        image = Image.open(str(path))
        img_tensor = self.transform(image)

        # print(path.name)
        class_name = path.name.split('.')[0]
        # print(class_name)
        class_name = class_name.split('_')[0][:-2]
        # print(class_name)
        class_id = self.class_name2id[class_name]

        label_tensor = torch.from_numpy(np.ascontiguousarray(class_id).astype('int64')).squeeze()

        return img_tensor, label_tensor


if __name__ == '__main__':
    # split_ucdataset_for_wgangp_and_classify('../UCMerced_LandUse/Images', image_size=256)

    # offline_aug_data('../UCMerced_LandUse/train64')

    dataset = UCDataset(image_dir='../UCMerced_LandUse/train64_tiny', train_aug=0, choose_classes=all_names,
                        img_size=64, use_crop=0)

    print(len(dataset))

    loader = DataLoader(dataset, batch_size=21, shuffle=True, num_workers=0)
    for i, batch_data in enumerate(loader):
        # print(len(batch_data))
        print(batch_data[0].shape, batch_data[1])
        if i % 10 == 0 and i > 0:
            print('Check done', i)
            break

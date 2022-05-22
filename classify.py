import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.optim as optim
from torch.optim import Adam

import os
import argparse

from resnet import *
from classify_utils import get_progress_bar, update_progress_bar, ApplyTransform

from uc_dataset import UCDataset

# 0. Define some parameters
parser = argparse.ArgumentParser(description='UCMerced Land Use')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# args.resume = True

max_epoch = 200
batch_size = 8
img_size = 64
train_aug = 1
gan_aug = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 1. Load and normalizing dataset
if gan_aug:
    train_dataset = UCDataset('../UCMerced_LandUse/train' + str(img_size) + 'withGAN', train_aug=train_aug)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
else:
    train_dataset = UCDataset('../UCMerced_LandUse/train' + str(img_size), train_aug=train_aug)
    train_dataset_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = UCDataset('../UCMerced_LandUse/test' + str(img_size))
test_dataset_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 2. Define a Convolutional Network
# net, model_name = LeNet(), 'LeNet'
net, model_name = ResNet18(), 'ResNet18'
# net, model_name = ResNet34(), 'ResNet34'
print(model_name + ' is ready!')
net = net.to(device)

save_dir = 'checkpoint' + '/' + model_name + '_bs' + str(batch_size) + '_is' + str(img_size) + '_aug' + str(
    train_aug) + '_gan' + str(gan_aug)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Use GPU or not
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    print("Let's use", torch.cuda.device_count(), "GPUs")
    cudnn.benchmark = True

start_epoch = 0
best_acc = 0

if args.resume == True:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_dir), 'Error : no checkpoint directory found!'
    checkpoint = torch.load(save_dir + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1

# 3. Define a loss function
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters(), lr=args.lr)


# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


# 4. Train the network on the training data

def epoch_train(epoch):
    running_loss = 0.0
    net.train()
    correct = 0
    total = 0
    progress_bar_obj = get_progress_bar(len(train_dataset_loader))
    print('Epoch', epoch, 'Train')
    for i, (inputs, labels) in enumerate(train_dataset_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # this line doesn't work when use cpu
        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        update_progress_bar(progress_bar_obj, index=i, loss=(running_loss / (i + 1)), acc=100. * (correct / total),
                            c=correct, t=total)


# 5. Test Network
def epoch_test(epoch):
    global best_acc
    net.eval()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataset_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print()
    print("Accuracy of whole test dataset: %.2f %%" % acc)

    if acc > best_acc:
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }

        torch.save(state, save_dir + '/ckpt.pth')
        best_acc = acc
        print('Acc > best_acc, Saving net, acc')


def do_train():
    for epoch in range(start_epoch, start_epoch + max_epoch):
        epoch_train(epoch)
        epoch_test(epoch)


def do_test():
    assert os.path.isdir(save_dir), 'Error : no checkpoint directory found!'
    checkpoint = torch.load(save_dir + '/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print('ckpt best acc:', best_acc)

    net.cuda()
    net.eval()

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_dataset_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    acc = 100 * correct / total
    print()
    print("Accuracy of whole test dataset: %.2f %%" % acc)


if __name__ == '__main__':
    do_train()

    do_test()

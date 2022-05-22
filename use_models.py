import shutil

import torch
from ACGAN import generator, discriminator
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from uc_dataset import mkdir, all_names
from pathlib import Path


# def dis_images(pkl_path, label):
#     D = discriminator(input_dim=3, output_dim=1, input_size=input_size, class_num=class_num)
#     D.load_state_dict(torch.load(pkl_path2))
#     D.cuda()
#     D.eval()
# 
#     with torch.no_grad():
#         samples = D(sample_z_, sample_y_)
# 
#         D_fake, C_fake = D(samples)
# 
#         pred = torch.max(C_fake, 1)[1]
#         label = torch.max(sample_y_, 1)[1]
#         print(pred.shape, label.shape, real_class)
#         print(label)
#         print(pred)

def gen_images(pkl_path, batch_size, real_class, save_dir, class_id2name):
    input_size = 64
    z_dim = 107
    class_num = 21

    mkdir(save_dir)

    G = generator(input_dim=z_dim, output_dim=3, input_size=input_size, class_num=class_num)
    G.load_state_dict(torch.load(pkl_path))
    G.cuda()
    G.eval()

    # print(G.state_dict())
    # for k,v in G.state_dict().items():
    #     print(k)
    # import sys
    # sys.exit()

    gen_label = torch.LongTensor([real_class])

    gen_label = gen_label.repeat(1, batch_size).permute(1, 0).squeeze(1)
    sample_y_ = torch.zeros((batch_size, class_num)).scatter_(1, gen_label.unsqueeze(1), 1)

    sample_z_ = torch.rand((batch_size, z_dim))

    sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

    # print(sample_z_.shape, sample_y_.shape)

    with torch.no_grad():
        samples = G(sample_z_, sample_y_)

    samples = (samples + 1) / 2

    for i, sample in enumerate(samples):
        sample = sample.permute(1, 2, 0)
        sample = np.array(sample.cpu())
        sample = np.clip(sample, 0, 1)
        # print(sample.shape, np.min(sample), np.max(sample), sample.dtype)
        sample = (sample * 255).astype('uint8')
        sample = Image.fromarray(sample)
        sample.save(save_dir + '/' + class_id2name[real_class] + 'XX' + '_GAN' + str(i) + '.jpg', quality=100)


def merge_dataset():
    new_dir = '../UCMerced_LandUse/train64withGAN'
    mkdir(new_dir)
    for i in Path('../UCMerced_LandUse/train64').glob('*.*'):
        shutil.copy(str(i), new_dir)

    for i in Path('../UCMerced_LandUse/train64_GAN').glob('*.*'):
        shutil.copy(str(i), new_dir)


if __name__ == '__main__':
    save_dir = '../UCMerced_LandUse/train64_GAN'

    pkl_path = './models_main2/uc/ACGAN/ACGAN_G_ep9.pkl'

    class_names = all_names.split(',')
    class_ids = [i for i in range(len(class_names))]
    class_name2id = dict(zip(class_names, class_ids))
    class_id2name = dict(zip(class_ids, class_names))

    for id in tqdm(class_ids):
        batch_size = 80
        real_class = id
        gen_images(pkl_path, batch_size, real_class, save_dir, class_id2name)

    merge_dataset()

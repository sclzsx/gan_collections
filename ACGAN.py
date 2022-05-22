import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
from uc_dataset import UCDataset, all_names


class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32, class_num=10):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            # nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        utils.initialize_weights(self)

    def forward(self, input, label):
        # print(self.input_dim + self.class_num)
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        # print(x.shape)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        # print(x.shape)
        x = self.deconv(x)

        return x


class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32, class_num=10):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            nn.Linear(1024, self.class_num),
        )
        utils.initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc1(x)
        d = self.dc(x)
        c = self.cl(x)

        return d, c


class ACGAN(object):
    def __init__(self, args):
        self.g_pkl_path = args.g_pkl_path
        self.d_pkl_path = args.d_pkl_path
        self.save_epoch_freq = args.save_epoch_freq
        self.datadir = args.datadir

        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = 107
        self.class_num = 21
        # self.z_dim = 62
        # self.class_num = 7
        # self.z_dim = 62
        # self.class_num = 10
        self.sample_num = self.class_num ** 2

        if self.g_pkl_path is not None:
            # args.lrG = args.lrG / 100
            args.lrG = args.lrG
        if self.d_pkl_path is not None:
            # args.lrD = args.lrD / 100
            args.lrG = args.lrG

        self.use_crop = args.use_crop
        self.train_aug = args.train_aug

        # load dataset
        self.data_loader = dataloader(dataset=self.dataset, input_size=self.input_size, batch_size=self.batch_size,
                                      split='train', train_aug=self.train_aug, use_crop=self.use_crop,
                                      datadir=self.datadir)
        data = self.data_loader.__iter__().__next__()[0]
        # print(data)

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size,
                           class_num=self.class_num)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size,
                               class_num=self.class_num)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lrD, betas=(args.beta1, args.beta2))

        # if self.g_pkl_path is not None:
        #     self.G_optimizer = optim.SGD(self.G.parameters(), lr=args.lrG, momentum=0.9, weight_decay=5e-4)
        # if self.d_pkl_path is not None:
        #     self.D_optimizer = optim.SGD(self.D.parameters(), lr=args.lrG, momentum=0.9, weight_decay=5e-4)

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.BCE_loss = nn.BCELoss().cuda()
            self.CE_loss = nn.CrossEntropyLoss().cuda()
        else:
            self.BCE_loss = nn.BCELoss()
            self.CE_loss = nn.CrossEntropyLoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise & condition
        self.sample_z_ = torch.zeros((self.sample_num, self.z_dim))
        for i in range(self.class_num):
            self.sample_z_[i * self.class_num] = torch.rand(1, self.z_dim)
            for j in range(1, self.class_num):
                self.sample_z_[i * self.class_num + j] = self.sample_z_[i * self.class_num]

        temp = torch.zeros((self.class_num, 1))
        for i in range(self.class_num):
            temp[i, 0] = i

        temp_y = torch.zeros((self.sample_num, 1))
        for i in range(self.class_num):
            temp_y[i * self.class_num: (i + 1) * self.class_num] = temp

        self.sample_y_ = torch.zeros((self.sample_num, self.class_num)).scatter_(1, temp_y.type(torch.LongTensor), 1)
        if self.gpu_mode:
            self.sample_z_, self.sample_y_ = self.sample_z_.cuda(), self.sample_y_.cuda()

    def train(self):
        if self.g_pkl_path is not None:
            self.G.load_state_dict(torch.load(self.g_pkl_path))
        if self.d_pkl_path is not None:
            self.D.load_state_dict(torch.load(self.d_pkl_path))

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        print_iter_freq = 0.1
        print_iter_freq = int(len(self.data_loader) * print_iter_freq)
        if print_iter_freq == 0:
            print_iter_freq = 1
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, y_) in enumerate(self.data_loader):
                # print(x_.shape, y_.shape, x_.dtype, y_.dtype, y_, torch.min(x_), torch.max(x_))
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                z_ = torch.rand((self.batch_size, self.z_dim))
                # print(y_.shape)
                y_vec_ = torch.zeros((self.batch_size, self.class_num)).scatter_(1,
                                                                                 y_.type(torch.LongTensor).unsqueeze(1),
                                                                                 1)
                if self.gpu_mode:
                    x_, z_, y_vec_ = x_.cuda(), z_.cuda(), y_vec_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real, C_real = self.D(x_)
                D_real_loss = self.BCE_loss(D_real, self.y_real_)
                C_real_loss = self.CE_loss(C_real, torch.max(y_vec_, 1)[1])

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                D_loss = D_real_loss + C_real_loss + D_fake_loss + C_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_, y_vec_)
                D_fake, C_fake = self.D(G_)

                G_loss = self.BCE_loss(D_fake, self.y_real_)
                C_fake_loss = self.CE_loss(C_fake, torch.max(y_vec_, 1)[1])

                G_loss += C_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iter + 1) % print_iter_freq) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          (
                              (epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size,
                              D_loss.item(),
                              G_loss.item()))

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

            if (epoch + 1) % self.save_epoch_freq == 0:
                with torch.no_grad():
                    self.visualize_results((epoch + 1))
                self.save(epoch)

                # # 也许是训练不好的问题根源！
                # self.data_loader = dataloader(dataset=self.dataset, input_size=self.input_size,
                #                               batch_size=self.batch_size,
                #                               split='train', train_aug=self.train_aug, use_crop=self.use_crop,
                #                               datadir=self.datadir)
                # vislabel = self.data_loader.__iter__().__next__()[1]
                # print(vislabel)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save(epoch)
        utils.generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch, self.save_epoch_freq)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))

        if fix:
            """ fixed noise """
            # print(self.sample_z_.shape, self.sample_y_.shape)
            samples = self.G(self.sample_z_, self.sample_y_)
        else:
            """ random noise """
            sample_y_ = torch.zeros(self.batch_size, self.class_num).scatter_(1, torch.randint(0, self.class_num - 1, (
                self.batch_size, 1)).type(torch.LongTensor), 1)
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_, sample_y_ = sample_z_.cuda(), sample_y_.cuda()

            samples = self.G(sample_z_, sample_y_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G_ep' + str(epoch) + '.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D_ep' + str(epoch) + '.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))

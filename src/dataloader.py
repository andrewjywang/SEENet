import os
import cv2
import numpy as np
from PIL import Image


class DataLoader(object):
    def __init__(self, real_path, flow_path, video_length, image_size_h=128, image_size_v=128, batch_size=1,
                 is_train=True):
        self.real_path = real_path
        self.flow_path = flow_path
        self.image_size_h = image_size_h
        self.image_size_v = image_size_v
        self.video_length = video_length
        self.batch_size = batch_size
        self.is_train = is_train
        self.real_dataset = {}
        self.real_items = []
        self.flow_dataset = {}
        self.flow_items = []
        self.batches = []

        self.read_file()
        self.obtain_items()
        self.check()
        # self.resize_images()

        self.index = 0
        self.index_list = np.array(range(len(self.flow_items)))

    def read_file(self):
        real_dataset = self.real_dataset
        flow_dataset = self.flow_dataset

        for i_a, action in enumerate(os.listdir(self.real_path)):
            real_dataset[action] = {}
            action_dir = os.path.join(self.real_path, action)
            for i_p, person in enumerate(os.listdir(action_dir)):
                real_dataset[action][person] = {}
                person_dir = os.path.join(action_dir, person)
                for i_s, scenario in enumerate(os.listdir(person_dir)):
                    real_dataset[action][person][scenario] = {}
                    scenario_dir = os.path.join(person_dir, scenario)
                    for i_c, clip in enumerate(os.listdir(scenario_dir)):
                        real_dataset[action][person][scenario][clip] = []
                        clip_dir = os.path.join(scenario_dir, clip)
                        for i_i, image in enumerate(os.listdir(clip_dir)):
                            image_file = os.path.join(clip_dir, image)
                            real_dataset[action][person][scenario][clip].append(image_file)
                        real_dataset[action][person][scenario][clip].sort()

        for i_a, action in enumerate(os.listdir(self.flow_path)):
            flow_dataset[action] = {}
            action_dir = os.path.join(self.flow_path, action)
            for i_p, person in enumerate(os.listdir(action_dir)):
                flow_dataset[action][person] = {}
                person_dir = os.path.join(action_dir, person)
                for i_s, scenario in enumerate(os.listdir(person_dir)):
                    flow_dataset[action][person][scenario] = {}
                    scenario_dir = os.path.join(person_dir, scenario)
                    for i_c, clip in enumerate(os.listdir(scenario_dir)):
                        flow_dataset[action][person][scenario][clip] = []
                        clip_dir = os.path.join(scenario_dir, clip)
                        for i_i, image in enumerate(os.listdir(clip_dir)):
                            image_file = os.path.join(clip_dir, image)
                            flow_dataset[action][person][scenario][clip].append(image_file)
                        flow_dataset[action][person][scenario][clip].sort()

    def obtain_items(self):
        print("Loading data ......")
        # data augment
        real_dataset = self.real_dataset
        flow_dataset = self.flow_dataset
        real_items = self.real_items
        flow_items = self.flow_items

        for action in flow_dataset:
            for person in flow_dataset[action]:
                for scenario in flow_dataset[action][person]:
                    for clip in flow_dataset[action][person][scenario]:
                        clip_flow_all = flow_dataset[action][person][scenario][clip]
                        clip_real_all = real_dataset[action][person][scenario][clip]
                        if len(clip_flow_all) > self.video_length:
                            label = str(action) + '_' + str(person) + '_' + str(scenario)
                            clip_real_imgs = []
                            for image_file in clip_real_all:
                                clip_real_imgs.append(image_file)
                            clip_flow_imgs = []
                            for image_file in clip_flow_all:
                                clip_flow_imgs.append(image_file)

                            if self.is_train:
                                for i in range(len(clip_real_imgs)):
                                    if (i + self.video_length) <= len(clip_real_imgs):
                                        real_item_list = clip_real_imgs[i:i + self.video_length]
                                        flow_item_list = clip_flow_imgs[i:i + self.video_length - 1]
                                        real_items.append((real_item_list, label))
                                        flow_items.append((flow_item_list, label))
                            else:
                                real_item_list = clip_real_imgs[0:self.video_length]
                                flow_item_list = clip_flow_imgs[0:self.video_length - 1]
                                real_items.append((real_item_list, label))
                                flow_items.append((flow_item_list, label))

    def read_image(self, path):
        try:
            img = Image.open(path)
            img = img.resize((self.image_size_h, self.image_size_v))
            img_m = np.asarray(img)
        except IOError:
            print('fail to load image!')

        if len(img_m.shape) != 3 or img_m.shape[2] != 3:
            print('Wrong image {} with shape {}'.format(path, img_m.shape))
            return None

        # range of pixel values = [-1.0, 1.0]
        img_m = img_m.astype(np.float32) / 255.0
        img_m = img_m * 2.0 - 1.0
        return img_m

    def shuffle_index_list(self):
        self.index = 0
        np.random.shuffle(self.index_list)

    def get_batch(self):
        _real_batch = []
        _flow_batch = []
        _labels = []
        _label = 0
        if (self.index + self.batch_size) > len(self.real_items) - 1:
            self.index = 0
        for i in range(self.batch_size):
            _labels.append(self.real_items[self.index_list[self.index]][1])
            _real_batch_files = self.real_items[self.index_list[self.index]][0]
            _flow_batch_files = self.flow_items[self.index_list[self.index]][0]
            _real_batch_imgs = []
            for image_file in _real_batch_files:
                img = self.read_image(image_file)
                _real_batch_imgs.append(img)
            _flow_batch_imgs = []
            for image_file in _flow_batch_files:
                img = self.read_image(image_file)
                _flow_batch_imgs.append(img)
            _real_batch.append(np.stack(_real_batch_imgs, 0))
            _flow_batch.append(np.stack(_flow_batch_imgs, 0))
            self.index = self.index + 1
        real_batch = np.stack(_real_batch, 0)
        flow_batch = np.stack(_flow_batch, 0)
        if _labels[0] == _labels[1]:
            _label = 1
        else:
            _label = 0
        return real_batch, flow_batch, _label

    def get_batch_num(self):
        return len(self.flow_items) // self.batch_size





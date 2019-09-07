import tensorflow as tf
import numpy as np
import os
import time
import random

from PIL import Image
from .utils import *
from .models import *


class SEENET(object):
    def __init__(self, args, sess, graph):
        self.epoch_num = args.epoch_num
        self.image_channel = args.image_channel
        self.image_shape = [args.image_size_v, args.image_size_h, args.image_channel]
        self.video_length = args.video_length
        self.pred_num = args.pred_num
        self.batch_size = args.batch_size
        self.motion_dim = args.motion_dim
        self.content_dim = args.content_dim
        self.num_layers = args.num_layers
        self.hidden_size = args.hidden_size
        self.log_dir = args.log_dir
        self.model_dir = args.pretrain_dir
        self.sample_dir = args.sample_dir

        self.video_inputs = tf.placeholder(tf.float32, [self.batch_size, self.video_length] + self.image_shape,
                                           name='video_inputs')
        self.flow_inputs = tf.placeholder(tf.float32, [self.batch_size, self.video_length - 1] + self.image_shape,
                                          name='flow_inputs')
        self.is_train = tf.placeholder(tf.bool, name='is_train')

        self.content_encoder = None
        self.motion_encoder = None
        self.content_decoder = None
        self.motion_decoder = None
        self.motion_lstmcell = None
        self.motion_output_layer = None
        self.motion_discriminator = None
        self.fusion_layer = None
        self.decoder = None
        self.discriminator = None

        self.loss_c = None
        self.loss_m = None
        self.loss_g = None
        self.loss_d = None
        self.optimizer_c = None
        self.optimizer_m = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.real_batch1_d = None
        self.real_batch2_d = None
        self.fake_batch1_d = None
        self.fake_batch2_d = None

        self.lr_c = args.lr_c
        self.lr_m = args.lr_m
        self.lr_g = args.lr_g
        self.lr_d = args.lr_d
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.delta = args.delta
        self.phi = args.phi
        self.chi = args.chi
        self.margin = args.margin
        self.same = 0
        self.real_label = args.real_label
        self.fake_label = args.fake_label
        self.train_feature = args.train_feature

        self.sess = sess
        self.graph = graph
        self.summary_op = None
        self.summary_op_g = None
        self.summary_op_d = None
        self.content_vars = []
        self.motion_vars = []
        self.decoder_vars = []
        self.g_vars = []
        self.d_vars = []
        self.all_vars = []
        self.pred_batch1 = None
        self.pred_batch2 = None
        self.psnrs = []
        self.ssims = []
        self.build()

    def build(self):
        self.content_encoder = ContentEncoder('content_encoder', self.is_train, self.content_dim)
        self.content_decoder = ContentDecoder('content_decoder', self.is_train, self.image_channel)
        self.motion_encoder = MotionEncoder('motion_encoder', self.is_train, self.motion_dim)
        self.motion_decoder = MotionDecoder('motion_decoder', self.is_train, self.image_channel)
        self.motion_lstmcell = MotionLSTM('motion_lstmcell', self.num_layers, self.hidden_size,
                                          self.motion_dim, self.video_length, self.pred_num, 1)
        self.motion_discriminator = MotionDiscriminator('motion_discriminator', self.num_layers, self.hidden_size,
                                                        self.video_length, 1)
        self.fusion_layer = Fusion('fusion_layer', self.content_dim)
        self.decoder = Decoder('decoder', self.is_train, self.image_channel)
        self.discriminator = Discriminator('discriminator', self.is_train)

        # build content loss
        # --- reconstruction loss
        # --- contrastive loss
        batch1_real = self.video_inputs[0, :, :, :, :]
        batch2_real = self.video_inputs[1, :, :, :, :]
        batch1_real_c = self.content_encoder(batch1_real)
        batch2_real_c = self.content_encoder(batch2_real)
        batch1_real_rec = self.content_decoder(batch1_real_c)
        batch2_real_rec = self.content_decoder(batch2_real_c)

        loss_real_rec_batch1 = tf.reduce_mean(tf.abs(batch1_real_rec - batch1_real))
        loss_real_rec_batch2 = tf.reduce_mean(tf.abs(batch2_real_rec - batch2_real))
        loss_real_rec = loss_real_rec_batch1 + loss_real_rec_batch2
        loss_contrastive, sim_dis, diff_dis = self.get_contrastive_loss(batch1_real_c, batch2_real_c)
        self.loss_c = self.alpha * loss_real_rec + self.beta * (loss_contrastive[0] + loss_contrastive[1])
        self.batch1_real_rec = batch1_real_rec
        sumc_1 = tf.summary.image('real_image_batch1', batch1_real, 1)
        sumc_2 = tf.summary.image('real_rec_image_batch1', batch1_real_rec, 1)
        sumc_3 = tf.summary.scalar('loss_real_rec', loss_real_rec)
        sumc_4 = tf.summary.scalar('loss_contrastive_sim', loss_contrastive[0])
        sumc_5 = tf.summary.scalar('loss_contrastive_diff', loss_contrastive[1])
        sumc_6 = tf.summary.histogram('sim_dis', sim_dis)
        sumc_7 = tf.summary.histogram('diff_dis', diff_dis)
        loss_c_sum = tf.summary.scalar('loss_c', self.loss_c)

        # build motion loss
        # --- motion latent code lstm loss
        # --- motion latent code lstm discriminator loss
        batch1_flow = self.flow_inputs[0, :, :, :, :]
        batch2_flow = self.flow_inputs[1, :, :, :, :]
        batch1_flow_m = self.motion_encoder(batch1_flow)
        batch2_flow_m = self.motion_encoder(batch2_flow)
        loss_lstm_batch1, batch1_flow_rec, batch1_xms = self.get_lstm_loss(batch1_flow, batch1_flow_m)
        loss_lstm_batch2, batch2_flow_rec, batch2_xms = self.get_lstm_loss(batch2_flow, batch2_flow_m)
        loss_lstm = loss_lstm_batch1 + loss_lstm_batch2
        batch1_flow_m_pred = tf.concat([batch1_flow_m[0:self.video_length - self.pred_num - 1, :], batch1_xms], 0)
        batch2_flow_m_pred = tf.concat([batch2_flow_m[0:self.video_length - self.pred_num - 1, :], batch2_xms], 0)
        loss_lstm_d, sums = self.get_lstm_discriminator_loss(batch1_flow_m_pred, batch2_flow_m_pred)
        self.loss_m = self.gamma * loss_lstm + self.delta * loss_lstm_d
        self.real_flow1 = batch1_flow[self.video_length - self.pred_num - 1:, :, :, :]
        self.real_flow1_rec = batch1_flow_rec
        summ_1 = tf.summary.image('flow_image', self.real_flow1, 20)
        summ_2 = tf.summary.image('flow_rec_image', batch1_flow_rec, 20)
        summ_3 = tf.summary.scalar('loss_lstm_d', loss_lstm_d)
        summ_4 = tf.summary.scalar('loss_lstm', loss_lstm)
        summ_5 = tf.summary.histogram('batch1_xms', batch1_xms[0])
        summ_6 = sums[0]
        summ_7 = sums[1]
        loss_m_sum = tf.summary.scalar('loss_m', self.loss_m)

        # optimize content model and motion model
        self.content_vars = self.content_encoder.var_list + self.content_decoder.var_list
        self.motion_vars = self.motion_encoder.var_list + self.motion_decoder.var_list + self.motion_lstmcell.var_list + \
                           self.motion_discriminator.var_list
        # self.motion_vars = self.motion_encoder.var_list + self.motion_decoder.var_list + self.motion_lstmcell.var_list
        self.optimizer_c = tf.train.AdamOptimizer(self.lr_c).minimize(self.loss_c, var_list=self.content_vars)
        self.optimizer_m = tf.train.AdamOptimizer(self.lr_m).minimize(self.loss_m, var_list=self.motion_vars)
        self.summary_op_c = tf.summary.merge(
            [loss_c_sum, sumc_3, sumc_4, sumc_5, sumc_6, sumc_7])
        self.summary_op_m = tf.summary.merge([loss_m_sum, summ_3, summ_4, summ_5, summ_6, summ_7])

        # build generator and discriminator
        # --- reconstruction loss
        # --- adversarial loss
        batch1_xcs = batch1_real_c[self.video_length - self.pred_num - 1:self.video_length - 1, :]
        batch2_xcs = batch2_real_c[self.video_length - self.pred_num - 1:self.video_length - 1, :]
        batch1_pred_reals = batch1_real[self.video_length - self.pred_num:, :, :, :]
        batch2_pred_reals = batch2_real[self.video_length - self.pred_num:, :, :, :]
        loss1_ae, batch1_recs = self.forward_pred(batch1_xcs, batch1_xms, batch1_pred_reals)
        loss2_ae, batch2_recs = self.forward_pred(batch2_xcs, batch2_xms, batch2_pred_reals)
        loss_ae = loss1_ae + loss2_ae

        real_batch1_d, real_batch1_d_logits = self.discriminator(batch1_pred_reals)
        fake_batch1_d, fake_batch1_d_logits = self.discriminator(batch1_recs)
        real_batch2_d, real_batch2_d_logits = self.discriminator(batch2_pred_reals)
        fake_batch2_d, fake_batch2_d_logits = self.discriminator(batch2_recs)

        real1_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_batch1_d_logits,
                                                    labels=tf.ones_like(real_batch1_d) * self.real_label))
        fake1_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_batch1_d_logits,
                                                    labels=tf.ones_like(fake_batch1_d) * self.fake_label))
        real2_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_batch2_d_logits,
                                                    labels=tf.ones_like(real_batch2_d) * self.real_label))
        fake2_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_batch2_d_logits,
                                                    labels=tf.ones_like(fake_batch2_d) * self.fake_label))
        loss_batch1_ad = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_batch1_d_logits, labels=tf.ones_like(fake_batch1_d)))
        loss_batch2_ad = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_batch2_d_logits, labels=tf.ones_like(fake_batch2_d)))
        loss_ad = loss_batch1_ad + loss_batch2_ad

        self.loss_d = real1_loss_d + fake1_loss_d + real2_loss_d + fake2_loss_d
        self.loss_g = self.chi * loss_ad + self.phi * loss_ae

        self.g_vars = self.decoder.var_list + self.fusion_layer.var_list
        self.d_vars = self.discriminator.var_list
        self.optimizer_g = tf.train.AdamOptimizer(self.lr_g).minimize(self.loss_g, var_list=self.g_vars)
        self.optimizer_d = tf.train.AdamOptimizer(self.lr_d).minimize(self.loss_d, var_list=self.d_vars)
        flow_images_sum = tf.summary.image('flow_images', batch1_flow_rec, 20)
        real_images_sum = tf.summary.image('real_images', batch1_pred_reals, 20)
        pred_images_sum = tf.summary.image('pred_images', batch1_recs, 20)
        real1_d_sum = tf.summary.histogram('real1_d_sum', real_batch1_d)
        fake1_d_sum = tf.summary.histogram('fake1_d_sum', fake_batch1_d)
        loss_batch1_ad_sum = tf.summary.scalar('loss_batch1_ad', loss_batch1_ad)
        loss_g_sum = tf.summary.scalar('loss_g', self.loss_g)
        loss_d_sum = tf.summary.scalar('loss_d', self.loss_d)
        self.summary_op_g = tf.summary.merge(
            [loss_g_sum, loss_batch1_ad_sum])
        self.summary_op_d = tf.summary.merge([loss_d_sum, real1_d_sum, fake1_d_sum])
        if self.train_feature:
            self.summary_op_imgs = tf.summary.merge([sumc_1, sumc_2, summ_1, summ_2])
        else:
            self.summary_op_imgs = tf.summary.merge([flow_images_sum, real_images_sum, pred_images_sum])
        self.all_vars = self.g_vars + self.d_vars + self.content_vars + self.motion_vars

        self.pred_batch1 = batch1_recs
        self.pred_batch2 = batch2_recs

        self.psnrs1 = self.get_psnr(batch1_recs, batch1_pred_reals)
        self.psnrs2 = self.get_psnr(batch2_recs, batch2_pred_reals)
        self.ssims1 = self.get_ssim(batch1_recs, batch1_pred_reals)
        self.ssims2 = self.get_ssim(batch2_recs, batch2_pred_reals)

    def forward_pred(self, x_cs, x_ms, x_reals):
        _x_c = tf.expand_dims(x_cs[0, :], 0)
        _ones = tf.ones([self.pred_num, 1])
        _x_cs = _ones * _x_c
        _x_f = self.fusion_layer([_x_cs, x_ms])
        rec = self.decoder(_x_f)
        loss = tf.reduce_mean(tf.abs(rec - x_reals))

        return loss, rec

    def get_contrastive_loss(self, batch1_real_c, batch2_real_c):
        batch1_real_c_l = batch1_real_c[::2, :]
        batch1_real_c_r = batch1_real_c[1::2, :]
        batch2_real_c_l = batch2_real_c[::2, :]
        batch2_real_c_r = batch2_real_c[1::2, :]

        loss1_sim, sim1_dis = self.contrastive_loss(batch1_real_c_l, batch1_real_c_r, 1, self.margin)
        loss2_sim, sim2_dis = self.contrastive_loss(batch2_real_c_l, batch2_real_c_r, 1, self.margin)
        loss1_diff, diff1_dis = self.contrastive_loss(batch1_real_c_l, batch2_real_c_r, self.same, self.margin)
        loss2_diff, diss2_dis = self.contrastive_loss(batch2_real_c_l, batch1_real_c_r, self.same, self.margin)
        sim_loss = loss1_sim + loss2_sim
        diff_loss = loss1_diff + loss2_diff
        return [sim_loss, diff_loss], sim1_dis, diff1_dis

    def contrastive_loss(self, left, right, y, margin, eps=1e-7):
        distance = tf.sqrt(eps + tf.reduce_sum(tf.square(left - right), 1))
        similarity = y * tf.square(distance)
        dissimilarity = (1 - y) * tf.square(tf.maximum((margin - distance), 0))
        return tf.reduce_mean(dissimilarity + similarity) / 2, distance

    def get_similarity_loss(self, batch1, batch2):
        batch1_list = tf.split(batch1, self.video_length, 0)
        batch2_list = tf.split(batch2, self.video_length, 0)
        loss1 = 0
        loss2 = 0
        for i in range(len(batch1_list) - 1):
            _loss1 = tf.reduce_mean(tf.square(batch1_list[i] - batch1_list[i + 1]))
            _loss2 = tf.reduce_mean(tf.square(batch2_list[i] - batch2_list[i + 1]))
            loss1 = loss1 + _loss1
            loss2 = loss2 + _loss2
        loss = loss1 + loss2
        return loss

    def get_lstm_loss(self, batch, batch_m):
        lstm_train_num = self.video_length - self.pred_num - 1
        outputs = self.motion_lstmcell(batch_m[0:lstm_train_num, :])
        flow_lstm_rec = self.motion_decoder(outputs)

        loss = tf.reduce_mean(tf.abs(flow_lstm_rec - batch[lstm_train_num:, :, :, :]))
        return loss, flow_lstm_rec, outputs

    def get_shuffle_list(self):
        shuffle_list = []
        shuffle_list_len = 0
        while shuffle_list_len < self.video_length - 1:
            random_number = random.randint(0, self.video_length - 1 - 1)
            if random_number not in shuffle_list:
                shuffle_list.append(random_number)
                shuffle_list_len = shuffle_list_len + 1
        return shuffle_list

    def get_lstm_discriminator_loss(self, batch1_flow_m, batch2_flow_m):
        batch1_flow_m_real = batch1_flow_m
        batch2_flow_m_real = batch2_flow_m
        batch1_flow_m_list = tf.split(batch1_flow_m, self.video_length - 1, 0)
        batch2_flow_m_list = tf.split(batch2_flow_m, self.video_length - 1, 0)
        batch1_flow_m_fake_list = []
        batch2_flow_m_fake_list = []
        shuffle_list = self.get_shuffle_list()

        for i in shuffle_list:
            batch1_flow_m_fake_list.append(batch1_flow_m_list[i])
            batch2_flow_m_fake_list.append(batch2_flow_m_list[i])

        batch1_flow_m_fake = tf.concat(batch1_flow_m_fake_list, 0)
        batch2_flow_m_fake = tf.concat(batch2_flow_m_fake_list, 0)
        real_batch1_d, real_batch1_d_logits = self.motion_discriminator(batch1_flow_m_real)
        fake_batch1_d, fake_batch1_d_logits = self.motion_discriminator(batch1_flow_m_fake)
        real_batch2_d, real_batch2_d_logits = self.motion_discriminator(batch2_flow_m_real)
        fake_batch2_d, fake_batch2_d_logits = self.motion_discriminator(batch2_flow_m_fake)

        real1_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_batch1_d_logits,
                                                    labels=tf.ones_like(real_batch1_d)))
        fake1_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_batch1_d_logits,
                                                    labels=tf.zeros_like(fake_batch1_d)))
        real2_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=real_batch2_d_logits,
                                                    labels=tf.ones_like(real_batch2_d)))
        fake2_loss_d = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_batch2_d_logits,
                                                    labels=tf.zeros_like(fake_batch2_d)))

        loss_d = real1_loss_d + fake1_loss_d + real2_loss_d + fake2_loss_d
        sum1 = tf.summary.histogram('real1_d_sum', real_batch1_d)
        sum2 = tf.summary.histogram('fake1_d_sum', fake_batch1_d)

        return loss_d, [sum1, sum2]

    def get_psnr(self, recs, reals):
        psnrs = []
        rec_list = tf.split(recs, self.pred_num, 0)
        real_list = tf.split(reals, self.pred_num, 0)
        for i in range(self.pred_num):
            im1 = tf.image.convert_image_dtype(rec_list[i], tf.float32)
            im2 = tf.image.convert_image_dtype(real_list[i], tf.float32)
            _psnr = tf.image.psnr(im1, im2, max_val=2.0)
            psnrs.append(_psnr)
        return tf.stack(psnrs)

    def get_ssim(self, recs, reals):
        ssims = []
        rec_list = tf.split(recs, self.pred_num, 0)
        real_list = tf.split(reals, self.pred_num, 0)
        for i in range(self.pred_num):
            im1 = tf.image.convert_image_dtype(rec_list[i], tf.float32)
            im2 = tf.image.convert_image_dtype(real_list[i], tf.float32)
            _ssim = tf.image.ssim(im1, im2, max_val=2.0)
            ssims.append(_ssim)
        return tf.stack(ssims)

    def train(self, loader, test_loader):
        prefix = time.strftime("%Y_%d_%m_%H_%M_%S", time.localtime())
        summary_dir = self.log_dir + prefix + '/'
        writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        params = ("SEENET"
                  + "_lr_c=" + str(self.lr_c)
                  + "_lr_m=" + str(self.lr_m)
                  + "_margin=" + str(self.margin)
                  + "_alpha=" + str(self.alpha)
                  + "_beta=" + str(self.beta)
                  + "_gamma=" + str(self.gamma)
                  + "_delta=" + str(self.delta))
        with open(summary_dir + '/params.txt', 'a') as file:
            file.write(params)

        init_all = tf.initializers.global_variables()
        self.sess.run(init_all)
        if self.train_feature:
            saver = tf.train.Saver(var_list=self.motion_vars + self.content_vars, max_to_keep=100)
            # saver.restore(self.sess, self.model_dir + 'SEENET_mc_new-6')
        else:
            saver0 = tf.train.Saver(var_list=self.motion_vars + self.content_vars, max_to_keep=100)
            saver0.restore(self.sess, self.model_dir + 'SEENET_mc_new-6')
            saver = tf.train.Saver(var_list=self.all_vars, max_to_keep=100)

        # Training epoch loop
        for epoch in range(self.epoch_num):
            print("In the epoch ", epoch)
            loader.shuffle_index_list()
            batch_num_all = loader.get_batch_num()
            for batch_num in range(batch_num_all):
                step = epoch * batch_num_all + batch_num
                video_batch, flow_batch, _same = loader.get_batch()
                print("In the epoch: " + str(epoch) + " and step: " + str(step) + " same: " + str(_same))
                self.same = _same
                if self.train_feature:
                    for j in range(1):
                        fetches = [self.loss_c, self.optimizer_c, self.summary_op_c]
                        fetched = self.sess.run(fetches, feed_dict={
                            'video_inputs:0': video_batch,
                            'flow_inputs:0': flow_batch,
                            'is_train:0': True
                        })
                        writer.add_summary(fetched[-1], step * 1 + j)
                    for k in range(3):
                        fetches = [self.loss_m, self.optimizer_m, self.summary_op_m]
                        fetched = self.sess.run(fetches, feed_dict={
                            'video_inputs:0': video_batch,
                            'flow_inputs:0': flow_batch,
                            'is_train:0': True
                        })
                        writer.add_summary(fetched[-1], step * 1 + k)
                else:
                    for j in range(1):
                        fetches_g = [self.psnrs, self.ssims, self.loss_g, self.optimizer_g, self.summary_op_g]
                        fetched_g = self.sess.run(fetches_g, feed_dict={
                            'video_inputs:0': video_batch,
                            'flow_inputs:0': flow_batch,
                            'is_train:0': True
                        })
                        writer.add_summary(fetched_g[-1], step * 1 + j)

                    for k in range(3):
                        fetches_d = [self.loss_d, self.optimizer_d, self.summary_op_d]
                        fetched_d = self.sess.run(fetches_d, feed_dict={
                            'video_inputs:0': video_batch,
                            'flow_inputs:0': flow_batch,
                            'is_train:0': True
                        })
                        writer.add_summary(fetched_d[-1], step * 3 + k)

                if step % 50 == 0:
                    fetches = [self.summary_op_imgs]
                    fetched = self.sess.run(fetches, feed_dict={
                        'video_inputs:0': video_batch,
                        'flow_inputs:0': flow_batch,
                        'is_train:0': True
                    })
                    writer.add_summary(fetched[-1], step)

            if self.train_feature:
                saver.save(self.sess, self.model_dir + 'SEENET_mc_new', global_step=epoch)
            else:
                saver.save(self.sess, self.model_dir + 'SEENET_all_new', global_step=epoch)
                # self.test_p(test_loader, epoch)

    def test_p(self, loader, epoch):
        for num in range(10):
            loader.shuffle_index_list()
            video_batch, flow_batch, _same = loader.get_batch()
            fetches_p = [self.pred_batch1, self.pred_batch2]
            fetched_p = self.sess.run(fetches_p, feed_dict={
                'video_inputs:0': video_batch,
                'flow_inputs:0': flow_batch,
                'is_train:0': False
            })
            folder_path = self.sample_dir + str(epoch) + '/' + str(num)
            folder = os.path.exists(folder_path)
            if not folder:
                os.makedirs(folder_path)
            _input_batch1 = np.split(video_batch[0, 0:self.video_length - self.pred_num, :, :, :],
                                     self.video_length - self.pred_num, 0)
            _input_batch2 = np.split(video_batch[1, 0:self.video_length - self.pred_num, :, :, :],
                                     self.video_length - self.pred_num, 0)
            _output_batch1 = np.split(fetched_p[0], self.pred_num, 0)
            _output_batch2 = np.split(fetched_p[1], self.pred_num, 0)
            for j in range(self.video_length - self.pred_num):
                self.save_image(j, _input_batch1[j], folder_path, 'input_batch1_')
                self.save_image(j, _input_batch2[j], folder_path, 'input_batch2_')
            for i in range(self.pred_num):
                self.save_image(i, _output_batch1[i], folder_path, 'output_batch1_')
                self.save_image(i, _output_batch2[i], folder_path, 'output_batch2_')

    def test_all(self, loader):
        saver = tf.train.Saver(var_list=self.all_vars)
        saver.restore(self.sess, self.model_dir + 'SEENET_all_new-5')
        loader.shuffle_index_list()
        for num in range(50):
            video_batch, flow_batch, _same = loader.get_batch()
            fetches_p = [self.pred_batch1, self.pred_batch2]
            fetched_p = self.sess.run(fetches_p, feed_dict={
                'video_inputs:0': video_batch,
                'flow_inputs:0': flow_batch,
                'is_train:0': False
            })
            folder_path = '../samples/' + str(num)
            folder = os.path.exists(folder_path)
            if not folder:
                os.makedirs(folder_path)
            _input_batch1 = np.split(video_batch[0, 0:self.video_length, :, :, :],
                                     self.video_length, 0)
            _input_batch2 = np.split(video_batch[1, 0:self.video_length, :, :, :],
                                     self.video_length, 0)
            _output_batch1 = np.split(fetched_p[0], self.pred_num, 0)
            _output_batch2 = np.split(fetched_p[1], self.pred_num, 0)
            for j in range(self.video_length):
                self.save_image(j, _input_batch1[j], folder_path, 'input_batch1_')
                self.save_image(j, _input_batch2[j], folder_path, 'input_batch2_')
            for i in range(self.pred_num):
                self.save_image(i, _output_batch1[i], folder_path, 'output_batch1_')
                self.save_image(i, _output_batch2[i], folder_path, 'output_batch2_')

    def save_image(self, idx, image, folder_path, suffix):
        img = np.squeeze(image, axis=0)
        img = Image.fromarray((((img + 1.0) / 2) * 255.0).astype('uint8'))
        path = folder_path + '/' + suffix + '_' + str(idx) + '.png'
        img.save(path)

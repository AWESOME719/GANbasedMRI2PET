# -*- coding: UTF-8 -*-
"""
    Name: Zheng TANG(Eng: Lional)
    Time: 2021/04/09
    Place: SIAT, Shenzhen
    Item: MRI --> PET
"""
from scipy.io import savemat
import os, datetime, shutil
import numpy as np
from utils import *
from models import *
import tensorflow as tf
from copy import deepcopy
from config import FLAGS
from datetime import datetime
import cv2
# from tensorflow_examples.models.pix2pix import pix2pix

"""==================== Configure ========================="""
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
GPUs = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(GPUs[0], True)
# tf.debugging.set_log_device_placement(True)
# VGG = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
"""======================= Main ============================"""


class CycleGAN():
    def __init__(self):
        # load dataset

        train_paths = glob_image_from_folder(FLAGS.train_dir)
        # print(train_paths)
        valid_paths = glob_image_from_folder(FLAGS.valid_dir)
        train_dataset = dataset_loader(train_paths, read_image_with_augment)
        valid_dataset = dataset_loader(valid_paths, read_image_without_augment)
        self.train_iter = iter(train_dataset)
        self.valid_iter = iter(valid_dataset)

        self.step_per_epoch = len(train_paths) // FLAGS.batch_size
        self.iteration = self.step_per_epoch * FLAGS.num_epoch
        # Build Two G and Two D model

        self.gen_g = GAN_G(FLAGS.img_size, FLAGS.img_ch, False, 256, name='G')
        self.gen_f = GAN_G(FLAGS.img_size, FLAGS.img_ch, False, 256, name='G')
        self.G_ema = deepcopy(self.gen_g)
        self.G_ema_f = deepcopy(self.gen_f)
        self.dis_x = GAN_D(FLAGS.img_size, True, 256, name='Discriminator')
        self.dis_y = GAN_D(FLAGS.img_size, True, 256, name='Discriminator')

        self.gen_g.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        self.gen_f.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        self.dis_x.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch * 2))
        self.dis_y.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch * 2))
        self.G_ema.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        self.G_ema_f.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        dis_x_params, dis_y_params, gen_g_params, gen_f_params = self.dis_x.count_params(), \
                                                                 self.dis_y.count_params(), \
                                                                 self.gen_g.count_params(), \
                                                                 self.gen_f.count_params()

        # Build each G and D optimizers

        gen_lr_schedule = LR_schedule(self.iteration, FLAGS.gen_lr, 'Piecewise')
        dis_lr_schedule = LR_schedule(self.iteration, FLAGS.dis_lr, 'Piecewise')
        self.gen_g_optimizer = tf.keras.optimizers.RMSprop(gen_lr_schedule)  # 利用RMSPROP优化算法效果更好
        self.gen_f_optimizer = tf.keras.optimizers.RMSprop(gen_lr_schedule)
        self.dis_x_optimizer = tf.keras.optimizers.RMSprop(dis_lr_schedule)
        self.dis_y_optimizer = tf.keras.optimizers.RMSprop(dis_lr_schedule)

        # Loss function
        self.BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.bce = tf.keras.losses.BinaryCrossentropy()
        # self.PCP = PCP(VGG).Content
        self.MAE = tf.keras.losses.MeanAbsoluteError()
        self.MSE = tf.keras.losses.MeanSquaredError()

        # Checkpoints
        self.ckpt, self.start_iteration = tf.train.Checkpoint(G_ema=self.G_ema), 0
        self.manager = tf.train.CheckpointManager(self.ckpt, FLAGS.logdir, max_to_keep=1)
        if FLAGS.finetune:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
            print('Latest checkpoint restored! start iteration: ', self.start_iteration)

        # Logging information
        print('*****************************************************************')
        print("Train Dataset number: ", len(train_paths))
        print("Valid Dataset number: ", len(valid_paths))
        print('Train %d Epoches (%d Iterations), %d Iteration per epoch'
              % (FLAGS.num_epoch, self.iteration, self.step_per_epoch))
        print('*****************************************************************')
        print("Gen_G network parameters: ", gen_g_params)
        print("Gen_F network parameters: ", gen_f_params)
        print("D_X network parameters: ", dis_x_params)
        print("D_Y network parameters: ", dis_y_params)
        print("Total network parameters: ", gen_g_params + gen_f_params + dis_x_params + dis_y_params)
        print('*****************************************************************')
        print('Initial G learning rate: %5g' % FLAGS.gen_lr)
        print('Initial D learning rate: %5g' % FLAGS.dis_lr)

    """================================= Loss ==================================="""

    def dis_loss(self, C_real, C_fake):
        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(C_real), logits=(C_real)))
        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(C_fake), logits=(C_fake)))
        D_loss = D_real_loss + D_fake_loss
        return D_loss

    def gen_gan_loss(self, fake_image):
        return self.BCE(tf.ones_like(fake_image), fake_image)

    def gen_pixel_loss(self, real_label, fake_image):
        # y_true = real_label * 0.5 + 0.5
        # y_pred = fake_image * 0.5 + 0.5
        # # loss = tf.reduce_sum([
        # #     1.00*self.bce(y_true, y_pred),
        # #     0.02*self.PCP(y_true, y_pred)])
        loss = self.MAE(real_label, fake_image)
        return 100 * loss

    def gen_cycle_loss(self, real_label, cycled_image):
        loss = self.MAE(real_label, cycled_image)
        return 10 * loss

    def gen_identity_loss(self, real_label, same_image):
        loss = self.MAE(real_label, same_image)
        return 5 * loss

    """=========================== Every Train Step ================================"""

    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape() as gen_tape, \
                tf.GradientTape() as gen_f_tape, \
                tf.GradientTape() as dis_x_tape, \
                tf.GradientTape() as dis_y_tape:
            # generator g: MRI -> PET;
            # discriminator_y: fake/real PET
            # generator_f: PET -> MRI
            # discriminator_x: fake/real MRI

            fake_PET = self.gen_g(image)
            fake_PET = tf.tanh(fake_PET)
            cycled_MRI = self.gen_f(fake_PET)
            cycled_MRI = tf.tanh(cycled_MRI)
            fake_MRI = self.gen_f(label)
            fake_MRI = tf.tanh(fake_MRI)
            cycled_PET = self.gen_g(fake_MRI)
            cycled_PET = tf.tanh(cycled_PET)

            # same_x and same_y for identity_loss
            same_MRI = self.gen_f(label)
            same_MRI = tf.tanh(same_MRI)
            same_PET = self.gen_g(image)
            same_PET = tf.tanh(same_PET)

            # Discriminator
            D_real_MRI = self.dis_x(tf.concat((image, label), -1))
            D_real_PET = self.dis_y(tf.concat((label, image), -1))
            D_fake_MRI = self.dis_x(tf.concat((fake_MRI, label), -1))
            D_fake_PET = self.dis_y(tf.concat((fake_PET, image), -1))

            # Calculate D loss
            disc_x_loss = self.dis_loss(D_real_MRI, D_fake_MRI)
            disc_y_loss = self.dis_loss(D_real_PET, D_fake_PET)

            # Total G loss = gen_gan_loss + cycle_loss + same_loss
            gen_g_loss = self.gen_gan_loss(D_fake_PET)
            gen_f_loss = self.gen_gan_loss(D_fake_MRI)
            pixel_g_loss = self.gen_pixel_loss(label, fake_PET)
            pixel_f_loss = self.gen_pixel_loss(image, fake_MRI)
            cycle_MRI_loss = self.gen_cycle_loss(image, cycled_MRI)
            cycle_PET_loss = self.gen_cycle_loss(label, cycled_PET)
            total_cycle_loss = cycle_MRI_loss + cycle_PET_loss
            same_umap_loss = self.gen_identity_loss(image, same_MRI)
            same_CT_loss = self.gen_identity_loss(label, same_PET)

            total_gen_g_loss = pixel_g_loss + gen_g_loss + \
                               total_cycle_loss + same_CT_loss
            total_gen_f_loss = pixel_f_loss + gen_f_loss + \
                               total_cycle_loss + same_umap_loss

            # Compute Metrics: PSNR, SSIM, PCC in self.generator_g
            # pix2pix.unet_generator predicts 'tanh' value.
            I = tf.clip_by_value(0.5 * (image + 1), 0, 1)
            LABEL = tf.clip_by_value(0.5 * (label + 1), 0, 1)
            FAKE_MRI = tf.clip_by_value(0.5 * (fake_MRI + 1), 0, 1)
            FAKE_PET = tf.clip_by_value(0.5 * (fake_PET + 1), 0, 1)
            psnr = PSNR(0.5 * fake_PET + 0.5, 0.5 * label + 0.5)
            ssim = SSIM(0.5 * fake_PET + 0.5, 0.5 * label + 0.5)
            pcc = PCC(0.5 * fake_PET + 0.5, 0.5 * label + 0.5)

            # Compute gradients。
        generator_g_gradients = gen_tape.gradient(total_gen_g_loss,
                                                  self.gen_g.trainable_variables)
        generator_f_gradients = gen_f_tape.gradient(total_gen_f_loss,
                                                    self.gen_f.trainable_variables)

        discriminator_x_gradients = dis_x_tape.gradient(disc_x_loss,
                                                        self.dis_x.trainable_variables)
        discriminator_y_gradients = dis_y_tape.gradient(disc_y_loss,
                                                        self.dis_y.trainable_variables)

        # Apply gradients to optimizer.
        self.gen_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                                 self.gen_g.trainable_variables))
        self.gen_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                                 self.gen_f.trainable_variables))
        self.dis_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                 self.dis_x.trainable_variables))
        self.dis_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                 self.dis_y.trainable_variables))
        return disc_y_loss, disc_x_loss, pixel_g_loss, gen_g_loss, cycle_MRI_loss, same_CT_loss, \
               total_gen_g_loss, total_gen_f_loss, pixel_f_loss, gen_f_loss, cycle_PET_loss, same_umap_loss, psnr, \
               ssim, pcc, I, LABEL, FAKE_PET, FAKE_MRI

    @tf.function
    def valid_step(self, image, label):
        logit = self.G_ema(image)
        logit = tf.tanh(logit)
        I = tf.clip_by_value(0.5 * (image + 1), 0, 1)
        F = tf.clip_by_value(0.5 * (logit + 1), 0, 1)
        L = tf.clip_by_value(0.5 * (label + 1), 0, 1)
        psnr = PSNR(0.5 * logit + 0.5, 0.5 * label + 0.5)
        ssim = SSIM(0.5 * logit + 0.5, 0.5 * label + 0.5)
        pcc = PCC(0.5 * logit + 0.5, 0.5 * label + 0.5)
        return psnr, ssim, pcc, I, F, L

    @tf.function
    def valid_step_CYCLE(self, image, label):
        logit = self.G_ema_f(image)
        logit = tf.tanh(logit)
        I = tf.clip_by_value(0.5 * (image + 1), 0, 1)
        F = tf.clip_by_value(0.5 * (logit + 1), 0, 1)
        L = tf.clip_by_value(0.5 * (label + 1), 0, 1)
        psnr = PSNR(0.5 * logit + 0.5, 0.5 * label + 0.5)
        ssim = SSIM(0.5 * logit + 0.5, 0.5 * label + 0.5)
        pcc = PCC(0.5 * logit + 0.5, 0.5 * label + 0.5)
        return psnr, ssim, pcc, I, F, L

    @tf.function
    def moving_average(self, model, model_test, beta=0.999):
        update_weight = model.trainable_weights
        previous_weight = model_test.trainable_weights
        for new_param, pre_param in zip(update_weight, previous_weight):
            average_param = beta * pre_param + (1 - beta) * new_param
            pre_param.assign(average_param)

    """============================ Update Training ==============================="""

    def train(self):
        self.summary_1 = tf.summary.create_file_writer(FLAGS.logdir + '/train')
        self.summary_2 = tf.summary.create_file_writer(FLAGS.logdir + '/valid')
        self.summary_3 = tf.summary.create_file_writer(FLAGS.logdir + '/valid_CYCLE')
        start_time = datetime.now()

        for idx in range(self.start_iteration, self.iteration):

            image, label = next(self.train_iter)
            label = -label

            disc_y_loss, disc_x_loss, pixel_g_loss, gen_g_loss, cycle_MRI_loss, same_CT_loss, total_gen_g_loss, \
            total_gen_f_loss, pixel_f_loss, gen_f_loss, cycle_PET_loss, same_umap_loss, \
            psnr, ssim, pcc, \
            I, LABEL, FAKE_PET, FAKE_MRI = self.train_step(image, label)

            self.moving_average(self.gen_g, self.G_ema)
            self.moving_average(self.gen_f, self.G_ema_f)
            epoch = idx // self.step_per_epoch
            # G_g: umap-->CT; D_y: real/fake CT
            # G_f: CT--> umap; D_x: real/fake umap
            print("[Epoch %3d, Iter %6d/%6d] [D_y/D_x loss: %.4g/%.4g] [G_g/G_f loss: %.4g/%.4g] "
                  "[PSNR/SSIM/PCC: %.2f/%.4f/%.4f] [time: %s]" % (epoch, idx, self.iteration,
                                                                  disc_y_loss, disc_x_loss, total_gen_g_loss,
                                                                  total_gen_f_loss, psnr,
                                                                  ssim, pcc, datetime.now() - start_time))

            if idx % self.step_per_epoch == 0:
                IMAGE_0, LABEL_0 = next(self.valid_iter)
                LABEL_0 = -LABEL_0
                # VALID STEP
                PSNR, SSIM, PCC, I_1, LOGIT, L_1 = self.valid_step(IMAGE_0, LABEL_0)
                PSNR_1, SSIM_1, PCC_1, I_2, LOGIT_1, L_2 = self.valid_step_CYCLE(LABEL_0, IMAGE_0)

                with self.summary_1.as_default():
                    tf.summary.scalar('Lr/Gen_G', self.gen_g_optimizer.lr(idx), epoch)
                    tf.summary.scalar('Lr/Gen_F', self.gen_f_optimizer.lr(idx), epoch)
                    tf.summary.scalar('Lr/D_X', self.dis_x_optimizer.lr(idx), epoch)
                    tf.summary.scalar('Lr/D_Y', self.dis_y_optimizer.lr(idx), epoch)
                    tf.summary.scalar('D_loss/x', disc_x_loss, epoch)  # real/fake umap
                    tf.summary.scalar('D_loss/y', disc_y_loss, epoch)  # real/fake CT
                    tf.summary.scalar('G_g/total_loss', total_gen_g_loss, epoch)
                    tf.summary.scalar('G_g/pixel_loss', pixel_g_loss, epoch)
                    tf.summary.scalar('G_g/gan_loss', gen_g_loss, epoch)
                    tf.summary.scalar('G_g/cycle_loss', cycle_MRI_loss, epoch)
                    tf.summary.scalar('G_g/same_loss', same_CT_loss, epoch)
                    tf.summary.scalar('G_f/total_loss', total_gen_f_loss, epoch)
                    tf.summary.scalar('G_f/pixel_loss', pixel_f_loss, epoch)
                    tf.summary.scalar('G_f/gan_loss', gen_f_loss, epoch)
                    tf.summary.scalar('G_f/cycle_loss', cycle_PET_loss, epoch)
                    tf.summary.scalar('G_f/same_loss', same_umap_loss, epoch)
                    tf.summary.scalar('metric/PSNR', psnr, epoch)
                    tf.summary.scalar('metric/SSIM', ssim, epoch)
                    tf.summary.scalar('metric/PCC', pcc, epoch)
                    tf.summary.image('PET/real', LABEL, epoch, 4)
                    tf.summary.image('PET/fake', FAKE_PET, epoch, 4)
                    tf.summary.image('MRI/real', I, epoch, 4)
                    tf.summary.image('MRI/fake', FAKE_MRI, epoch, 4)

                with self.summary_2.as_default():
                    tf.summary.scalar('Metrics/PSNR', PSNR, epoch)
                    tf.summary.scalar('Metrics/SSIM', SSIM, epoch)
                    tf.summary.scalar('Metrics/PCC', PCC, epoch)
                    tf.summary.image('PET/real', L_1, epoch, 4)
                    tf.summary.image('PET/fake', LOGIT, epoch, 4)
                    tf.summary.image('MRI/real', I_1, epoch, 4)

                with self.summary_3.as_default():
                    tf.summary.scalar('Metrics/PSNR', PSNR_1, epoch)
                    tf.summary.scalar('Metrics/SSIM', SSIM_1, epoch)
                    tf.summary.scalar('Metrics/PCC', PCC_1, epoch)
                    tf.summary.image('MRI/fake', LOGIT_1, epoch, 4)
                    tf.summary.image('MRI/real', L_2, epoch, 4)
                    tf.summary.image('PET/real', I_2, epoch, 4)

                # save the model after training the generator reloaded in the future
                if epoch % 10 == 0:
                    self.manager.save(checkpoint_number=idx)

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    """============================ inference =============================="""

    # saveImage = tf.keras.preprocessing.image.save_img
    def test(self):
        self.save_dir = os.path.join('./outs', FLAGS.logdir[7:])
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # Restore models
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')
        psnr, ssim, mae = [], [], []
        test_paths = glob_image_from_folder(FLAGS.test_dir)

        for i, path in enumerate(test_paths):
            print(path)
            image, label = read_image_without_augment(path)
            label = -label
            I = tf.concat([image], -1)
            logit = self.G_ema(I[None])[0]
            logit = tf.tanh(logit[:, :, :1])
            image = tf.clip_by_value(image * 32767 + 32767, 0, 65535)
            label = label * 6000 + 6000
            logit = logit * 6000 + 6000
            combined = np.concatenate((image, logit, label), 1)

            mae.append(tf.reduce_mean(tf.abs(label - logit)))
            psnr.append(tf.reduce_mean(tf.image.psnr(label, logit, 65535)))
            ssim.append(tf.reduce_mean(tf.image.ssim(label, logit, 65535)))

            save_name = self.save_dir + '/%04d.png' % i
            cv2.imwrite(save_name, combined.astype(np.uint16))
        print('%s,MAE:%%.4f,PSNR:%.2fdB,SSIM:%.2f%%'
              % (sum(mae)/len(mae), sum(psnr)/len(psnr), sum(ssim)/len(ssim)))
        dicts = {"psnr": psnr, "ssim": ssim, 'mae': mae}
        savemat(self.save_dir + '/metrics.mat', dicts)
        save_csv(self.save_dir + '/metrics.csv', {
            'psnr': psnr, 'ssim': ssim, 'mae': mae})
        save_file = self.save_dir + '/results.txt'
        fprint(save_file, str(datetime.now()))
        fprint(save_file, '==================================================')
        fprint(save_file, 'For MRI->PET: ')
        fprint(save_file, 'PSNR = %.4f +/- %.4f dB' % list_mean_std(psnr, 1))
        fprint(save_file, 'SSIM = %.4f +/- %.4f %%' % list_mean_std(ssim, 100))
        fprint(save_file, 'MAE = %.4f +/- %.4f HU' % list_mean_std(mae, 1))
        # fprint(save_file, 'PCC = %.4f +/- %.4f HU' % list_mean_std(pcc, 1))
        fprint(save_file, '==================================================')


"""============================ RUN ==============================="""

if __name__ == '__main__':
    run_on_gpu()
    check_logdir()
    gan = CycleGAN()
    if FLAGS.phase == 'train':
        gan.train()
    gan.test()

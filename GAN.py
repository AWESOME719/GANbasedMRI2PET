#-*- coding: UTF-8 -*-
from scipy.io import savemat
from datetime import datetime
from copy import deepcopy
import tensorflow as tf
import cv2
import numpy as np
from config import FLAGS
from utils import *
from models import *
import matplotlib.pyplot as plt
"""================================= Main ==================================="""
class GAN():
    def __init__(self):
        # Data Loader
        train_paths = glob_image_from_folder(FLAGS.train_dir)
        # print(train_paths)
        valid_paths = glob_image_from_folder(FLAGS.valid_dir)
        train_dataset = dataset_loader(train_paths, read_image_with_augment)
        valid_dataset = dataset_loader(valid_paths, read_image_without_augment)
        self.train_iter = iter(train_dataset)
        self.valid_iter = iter(valid_dataset)

        self.step_per_epoch = len(train_paths) // FLAGS.batch_size
        self.iteration = self.step_per_epoch * FLAGS.num_epoch

        # Build Model
        self.G = GAN_G(FLAGS.img_size, FLAGS.img_ch, False, 256, name='G')
        self.D = GAN_D(FLAGS.img_size, True, 256, name='Discriminator')
        self.G_ema = deepcopy(self.G)

        self.D.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch*2))
        self.G.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        self.G_ema.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        D_params, G_params = self.D.count_params(), self.G.count_params()

        # Optimizer
        g_lr_schedule = LR_schedule(self.iteration, FLAGS.gen_lr, 'Piecewise')
        d_lr_schedule = LR_schedule(self.iteration, FLAGS.dis_lr, 'Piecewise')
        self.G_optimizer = tf.keras.optimizers.RMSprop(g_lr_schedule)#利用RMSPROP优化算法效果更好
        self.D_optimizer = tf.keras.optimizers.RMSprop(d_lr_schedule)
        self.gen_tv = self.G.trainable_variables
        self.dis_tv = self.D.trainable_variables
        # # ADD GAN-LOSS-WEIGHT
        # self.gan_loss_weight = 0.0

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
              %(FLAGS.num_epoch, self.iteration, self.step_per_epoch))
        print('*****************************************************************')
        print("G network parameters: ", G_params)
        print("D network parameters: ", D_params)
        print("Total network parameters: ", G_params + D_params)
        print('*****************************************************************')
        print('Initial G learning rate: %5g'% FLAGS.gen_lr)
        print('Initial D learning rate: %5g'% FLAGS.dis_lr)

    """================================= Loss ==================================="""
    def dis_gan_loss(self, C_real, C_fake):
        D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(C_real), logits=(C_real)))
        D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(C_fake), logits=(C_fake)))
        D_loss = D_real_loss + D_fake_loss

        # D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #             labels=tf.ones_like(C_real), logits=(C_real-C_fake)))
        return 1.0 * D_loss

    def dis_reg_loss(self):
        reg_loss = tf.nn.scale_regularization_loss(self.D.losses)
        return  reg_loss

    def gen_gan_loss(self, C_fake, C_real):
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(C_fake), logits=(C_fake)))

        # D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #             labels=tf.ones_like(C_fake), logits=(C_fake-C_real)))
        # return self.gan_loss_weight * D_loss
        # D_loss = self.gan_loss_weight * G_loss
        return  G_loss

    def gen_mae_loss(self, real, fake):
        mae_loss = tf.reduce_mean(tf.abs(real-fake))
        return 1.0 * mae_loss

    def gen_reg_loss(self):
        reg_G = tf.nn.scale_regularization_loss(self.G.losses)
        return  reg_G

    """=========================== Every Train Step ================================"""
    @tf.function
    def train_step(self, image, label):
        with tf.GradientTape(persistent=True) as gen_tape,\
                tf.GradientTape() as dis_tape:
            # input: [PET, MRI], B, H, W, 2*C
            # logit: [MRI, PET], B, H, W, 2*C
            logit = self.G(image)
            logit = tf.tanh(logit)

            C_fake = self.D(tf.concat((logit, image), -1))
            C_real = self.D(tf.concat((label, image), -1))

            d_gan_loss = self.dis_gan_loss(C_real, C_fake)
            # d_grad_loss= self.dis_grad_loss(logit, label)
            d_reg_loss = self.dis_reg_loss()
            d_loss = d_gan_loss + d_reg_loss

            g_gan_loss = self.gen_gan_loss(C_fake, C_real)
            g_mae_loss = self.gen_mae_loss(label, logit)
            g_reg_loss = self.gen_reg_loss()
            g_loss = g_gan_loss*self.gan_loss_weight + g_mae_loss + g_reg_loss

            # Metrics
            I = tf.clip_by_value(0.5 * (image + 1), 0, 1)#从-1/+1映射到0~1
            F = tf.clip_by_value(0.5 * (logit + 1), 0, 1)
            L = tf.clip_by_value(0.5 * (label + 1), 0, 1)
            psnr = tf.reduce_mean(tf.image.psnr(F, L, 1.0))
            ssim = tf.reduce_mean(tf.image.ssim(F, L, 1.0))
            mae = tf.reduce_mean(tf.abs(F-L))

        g_gradient = gen_tape.gradient(g_loss, self.gen_tv)
        d_gradient = dis_tape.gradient(d_loss, self.dis_tv)
        self.G_optimizer.apply_gradients(zip(g_gradient, self.gen_tv))
        self.D_optimizer.apply_gradients(zip(d_gradient, self.dis_tv))

        return g_loss, g_gan_loss, g_mae_loss, \
               g_reg_loss, \
               d_loss, d_gan_loss, d_reg_loss, \
               tf.reduce_mean(tf.sigmoid(C_real)),\
               tf.reduce_mean(tf.sigmoid(C_fake)), \
               psnr, ssim, mae, I, L, F

    @tf.function
    def valid_step(self, image, label):
        logit = self.G_ema(image)
        logit = tf.tanh(logit)

        I = tf.clip_by_value(0.5 * (image + 1), 0, 1)
        F = tf.clip_by_value(0.5 * (logit + 1), 0, 1)
        L = tf.clip_by_value(0.5 * (label + 1), 0, 1)
        psnr = tf.reduce_mean(tf.image.psnr(F, L, 1.0))
        ssim = tf.reduce_mean(tf.image.ssim(F, L, 1.0))
        mae = tf.reduce_mean(tf.abs(F - L))
        return psnr, ssim, mae, I, L, F

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
        start_time = datetime.now()
        for idx in range(self.start_iteration, self.iteration):

            # Todo: Add dynamic gan_loss_weight
            self.gan_loss_weight = 2.0 * max(idx / self.iteration - 0.25, 0)

            image, label = next(self.train_iter)
            label = -label
            # PET-MRI
            g_loss, g_gan_loss, g_mae_loss, \
            g_reg_loss,\
            d_loss, d_gan_loss, d_reg_loss, \
            C_real, C_fake, psnr, ssim, mae, I, L, F =  self.train_step(image, label)

            self.moving_average(self.G, self.G_ema)

            epoch = idx // self.step_per_epoch
            print("[Epoch %3d, Iter %6d/%6d] [D: %.4f, C: %.4f/%.4f]"
                  " [G: %.4f/%.4f][PSNR/SSIM/MAE: %.4g/%.4g/%.4g][time:%s]"
                  % (epoch, idx, self.iteration, d_loss, C_real, C_fake,
                     g_gan_loss, g_loss, psnr, ssim*100, mae*100,
                     datetime.now() - start_time))

            if idx % self.step_per_epoch ==0:
                IMAGE, LABEL = next(self.valid_iter)
                LABEL = -LABEL
                # print(IMAGE.shape)
                PSNR, SSIM, MAE, IMAGE, LABEL, LOGIT = self.valid_step(IMAGE, LABEL)

                with self.summary_1.as_default():
                    tf.summary.scalar('Lr/G', self.G_optimizer.lr(idx), epoch)
                    tf.summary.scalar('Lr/D', self.D_optimizer.lr(idx), epoch)
                    tf.summary.scalar('D/All_loss', d_loss, step=epoch)
                    tf.summary.scalar('D/Gan_loss', d_gan_loss, step=epoch)
                    tf.summary.scalar('D/Reg_loss', d_reg_loss, step=epoch)
                    # tf.summary.scalar('D/Grad_loss', d_grad_loss, step=epoch)
                    tf.summary.scalar('D/C_real', C_real, step=epoch)
                    tf.summary.scalar('D/C_fake', C_fake, step=epoch)
                    tf.summary.scalar('G/All_loss', g_loss, step=epoch)
                    tf.summary.scalar('G/Gan_loss_weight',self.gan_loss_weight, step=epoch)
                    tf.summary.scalar('G/Gan_loss', g_gan_loss, step=epoch)
                    tf.summary.scalar('G/Mae_loss', g_mae_loss, step=epoch)
                    # tf.summary.scalar('G/Pcp_loss', g_pcp_loss, step=epoch)
                    tf.summary.scalar('G/Reg_loss', g_reg_loss, step=epoch)
                    tf.summary.scalar('Metrics/PSNR', psnr, step=epoch)
                    tf.summary.scalar('Metrics/SSIM', ssim, step=epoch)
                    tf.summary.scalar('Metrics/MAE',   mae, step=epoch)
                    tf.summary.image('Image', I, epoch, 4)
                    tf.summary.image('Label', L, epoch, 4)
                    tf.summary.image('Logit', F, epoch, 4)

                with self.summary_2.as_default():
                    tf.summary.scalar('Metrics/PSNR', PSNR, epoch)
                    tf.summary.scalar('Metrics/SSIM', SSIM, epoch)
                    tf.summary.scalar('Metrics/MAE',  MAE, epoch)
                    tf.summary.image('Image', IMAGE, epoch, 4)
                    tf.summary.image('Label', LABEL, epoch, 4)
                    tf.summary.image('Logit', LOGIT, epoch, 4)

                # save every self.save_freq
                if epoch % 10 == 0:
                    self.manager.save(checkpoint_number=idx)

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    """============================ Testing and Results ==============================="""
    def test(self):
        self.save_dir = os.path.join('./outs', FLAGS.logdir[7:])
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        # Restore models
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')


        test_paths = glob_image_from_folder(FLAGS.test_dir)
        # test_paths.extend(glob_image_from_folder(FLAGS.test_dir + '/coronal'))
        # test_paths.extend(glob_image_from_folder(FLAGS.test_dir + '/sagittal'))

        psnr, ssim, mae = [], [], []
        for i, path in enumerate(test_paths):
            print(path)

            image, label = read_image_without_augment(path)
            label = -label
            # patch_size = (FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch)
            # patch = tf_split_to_patches(image, patch_size, FLAGS.img_size)
            # patches = tf.reshape(patch, (-1,) + patch_size) #[B,256,256,1]
            # pMRI = self.G_ema(patches)[0]
            I = tf.concat([image],-1)
            # logit = self.G_ema(I[None])[0]
            # logit = tf.tanh(logit[:,:,:1])
            logit = tf.tanh(self.G_ema(image[None])[0])
            # pMRI_patch = tf.reshape(pMRI, patch.shape)
            # logit = tf_merge_from_patches(pMRI_patch, label.shape)
            # logit = self.G_ema(image)[0]
            # pMRI = self.G_ema(image)
            # pMRI_patch = tf.reshape(pMRI, patch.shape)
            # logit = tf_merge_from_patches(pMRI, label.shape)

            # normalize to [0, 1]
            # image = tf.clip_by_value(image, -32768, 32767)
            # label = tf.clip_by_value(label * 3000 + 3000, 0, 65535)
            # logit = tf.clip_by_value(logit, -32768, 32767)
            image = tf.clip_by_value(image*32767 + 32767, 0, 65535)
            label = tf.clip_by_value(label*6000 + 6000, 0, 65535)
            logit = tf.clip_by_value(logit*6000 + 6000, 0, 65535)
            # logit = tf.clip_by_value(logit*32767 + 32767, 0, 65535)
            # logit = -logit
            # label = -label
            # image = tf.clip_by_value(image*32767+32767, 0, 65535)
            # label = tf.clip_by_value(label*32767+32767, 0, 65535)
            # logit = tf.clip_by_value(logit*32767+32767, 0, 65535)
            psnr.append(tf.reduce_mean(tf.image.psnr(label, logit, 65535)))
            ssim.append(tf.reduce_mean(tf.image.ssim(label, logit, 65535)))
            mae.append(tf.reduce_mean(tf.abs(label - logit)))
            # label = -label
            comb = np.concatenate((image, label, logit), 1)
            # print(comb.dtype)
            # comb = tf.cast(comb,dtype=tf.uint16)
            save_name = self.save_dir + '/%04d.png'%i
            # save_name_1 = self.save_dir + '/logit' + '/%04d.jpg'%i
            # save_name_2 = self.save_dir + '/label' + '/%04d.jpg'%i
            # encoded_image = tf.image.encode_png(comb)
            # save_name_1 = self.save_dir + '/%04d_out.jpg'%i
            # tf.keras.preprocessing.image.save_img(save_name, comb)
            # cv2.imwrite(save_name, comb.astype(np.uint16))
            cv2.imwrite(save_name, comb.astype(np.uint16))
            # tf.keras.preprocessing.image.save_img(save_name_1, logit)
            # tf.keras.preprocessing.image.save_img(save_name_2, label)
            # tf.keras.preprocessing.image.save_img(save_name_1, logit)
        print('PSNR=%.2fdB, MAE=%.2f%%, SSIM=%.2f%%' % (sum(psnr)/len(psnr),
                sum(mae)/len(mae)*100, sum(ssim)/len(ssim)*100))
        dicts = {"psnr": psnr, "ssim": ssim, 'mae': mae}
        savemat(self.save_dir + '/metrics.mat', dicts)
        save_csv(self.save_dir + '/metrics.csv',{
            'psnr': psnr, 'ssim': ssim, 'mae': mae})
        save_file = self.save_dir + '/results.txt'
        fprint(save_file, str(datetime.now()))
        fprint(save_file, '==================================================')
        fprint(save_file, 'For MRI->PET: ')
        fprint(save_file, 'PSNR = %.4f +/- %.4f dB' % list_mean_std(psnr, 1))
        fprint(save_file, 'SSIM = %.4f +/- %.4f %%' % list_mean_std(ssim, 100))
        fprint(save_file, 'MAE = %.4f +/- %.4f HU' % list_mean_std(mae, 1))
        fprint(save_file, '==================================================')
"""============================ RUN ==============================="""
if __name__ == '__main__':
    run_on_gpu()
    check_logdir()
    gan = GAN()
    if FLAGS.phase == 'train':
        gan.train()
    gan.test()




#-*- coding: UTF-8 -*-
"""
    Name: Zheng TANG
    Time: 2021/03/28
    Place: SIAT, Shenzhen
    Item: U-NET
"""
from datetime import datetime
from copy import deepcopy
import tensorflow as tf
import numpy as np, cv2

from config import FLAGS
from utils import *
from models import *

"""================================= Main ==================================="""
class MultiCNN():
    def __init__(self):
        # Data Loader
        # if FLAGS.phase == 'test':
        #     data = np.load('./data/' + FLAGS.logdir[7:] + '.npz')
        #     train_data = list(data['train'])
        #     valid_data = list(data['valid'])
        #     infer_data = list(data['infer'])
        # else:
        #     paths = glob_image_from_folder(FLAGS.datadir)
        #     train_data, valid_data, infer_data = split_dataset(paths)
        #     train_data = glob_image_from_folder(FLAGS.datadir + '/train')
        #     train_paths = glob_image_from_folder(FLAGS.train_dir)
        #     valid_paths = glob_image_from_folder(FLAGS.valid_dir)
        #     infer_data = glob_image_from_folder(FLAGS.datadir + '/infer')
        #     np.savez('./data/' + FLAGS.logdir[7:] + '.npz',
        #              train=train_data, valid=valid_data, infer=infer_data)
        #-----------------------------------------------------------------------
        # train_dataset = dataset_loader(train_data, read_jpg_with_augment)
        # valid_dataset = dataset_loader(valid_data, read_jpg_with_augment)
        train_data = glob_image_from_folder(FLAGS.train_dir)
        valid_data = glob_image_from_folder(FLAGS.valid_dir)
        train_dataset = dataset_loader(train_data, read_image_with_augment)
        valid_dataset = dataset_loader(valid_data, read_image_without_augment)
        self.train_iter = iter(train_dataset)
        self.valid_iter = iter(valid_dataset)
        # self.test_data = infer_data

        self.step_per_epoch = len(train_data) // FLAGS.batch_size
        self.iteration = self.step_per_epoch * FLAGS.num_epoch

        # Build Model
        # self.G = UNet_AE(FLAGS.img_size, 2*FLAGS.img_ch, False, 512, name='G')
        self.G = UNet_AE(FLAGS.img_size, FLAGS.img_ch, False, 512, name='G')
        self.G_ema = deepcopy(self.G)
        self.G.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        self.G_ema.build((None, FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch))
        G_params = self.G.count_params()

        # Optimizer
        g_lr_schedule = LR_schedule(self.iteration, FLAGS.gen_lr, 'Piecewise')
        self.G_optimizer = tf.keras.optimizers.RMSprop(g_lr_schedule)
        self.gen_tv = self.G.trainable_variables

        # PCP set
        self.VGG19 = tf.keras.applications.VGG19(include_top=False,
                                                 weights='imagenet')
        self.VGG_layers = ['block1_pool','block2_pool','block3_pool',
                           'block4_pool','block5_pool']

        # Checkpoints
        self.ckpt, self.start_iteration = tf.train.Checkpoint(G_ema=self.G_ema), 0
        self.manager = tf.train.CheckpointManager(self.ckpt, FLAGS.logdir, max_to_keep=1)
        if FLAGS.finetune:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
            print('Latest checkpoint restored! start iteration: ', self.start_iteration)

        # # Save dirs
        # self.save_dir = os.path.join('./outs', FLAGS.logdir[7:])

        # Logging information
        print('*****************************************************************')
        print("Train data number: ", len(train_data))
        print("Valid data number: ", len(valid_data))
        # print("Test data number: ", len(self.test_data))
        print('Train %d Epoches (%d Iterations), %d Iteration per epoch'
              %(FLAGS.num_epoch, self.iteration, self.step_per_epoch))
        print('*****************************************************************')
        print("G network parameters: ", G_params)
        print('*****************************************************************')
        print('Initial G learning rate: %5g'% FLAGS.gen_lr)

    """================================= Loss ==================================="""
    def gen_mae_loss(self, real_label, fake_image):
        mae_loss = tf.reduce_mean(tf.abs(real_label-fake_image))
        return 1.0 * mae_loss

    def gen_pcp_loss(self, real, fake):
        pcp_loss = PCP(self.VGG19, self.VGG_layers).Loss(real, fake)
        return 1.0 * pcp_loss

    def gen_reg_loss(self):
        # Scales the sum of the given regularization losses
        # by number of replicas (In tf.distribute.Strategy) .
        reg_G = tf.nn.scale_regularization_loss(self.G.losses)
        return 1.0 * reg_G

    """=========================== Every Train Step ================================"""
    @tf.function
    # def train_step(self, image, label1, label2):
    def train_step(self, image, label):
        with tf.GradientTape(persistent=True) as gen_tape:
            # image = -image
            # label = -label
            logit = tf.tanh(self.G(image))
            # fake1, fake2 = tf.split(logit, 2, -1)
            fake = logit

            g_mae_loss = self.gen_mae_loss(label, fake)
            #              + self.gen_mae_loss(label2, fake2)
            # g_mae_loss = self.gen_mae_loss(label1, fake1) \
            #              + self.gen_mae_loss(label2, fake2)
            # g_pcp_loss = self.gen_pcp_loss(label1, fake1) \
            #              + self.gen_pcp_loss(label2, fake2)
            g_pcp_loss = self.gen_pcp_loss(label, fake) \
                         # + self.gen_pcp_loss(label2, fake2)
            g_reg_loss = self.gen_reg_loss()
            g_loss = g_mae_loss + g_pcp_loss + g_reg_loss

            # Metrics
            I = tf.clip_by_value(0.5 * (image + 1), 0, 1)
            # F1 = tf.clip_by_value(0.5 * (fake1 + 1), 0, 1)
            F1 = tf.clip_by_value(0.5 * (fake + 1), 0, 1)
            # F2 = tf.clip_by_value(0.5 * (fake2 + 1), 0, 1)
            L1 = tf.clip_by_value(0.5 * (label + 1), 0, 1)
            # L1 = tf.clip_by_value(0.5 * (label1 + 1), 0, 1)
            # L2 = tf.clip_by_value(0.5 * (label2 + 1), 0, 1)
            psnr1 = tf.reduce_mean(tf.image.psnr(F1, L1, 1.0))
            ssim1 = tf.reduce_mean(tf.image.ssim(F1, L1, 1.0))
            mae1 = tf.reduce_mean(tf.abs(F1 - L1))
            # psnr2 = tf.reduce_mean(tf.image.psnr(F2, L2, 1.0))
            # ssim2 = tf.reduce_mean(tf.image.ssim(F2, L2, 1.0))
            # mae2 = tf.reduce_mean(tf.abs(F2 - L2))

        g_gradient = gen_tape.gradient(g_loss, self.gen_tv)
        self.G_optimizer.apply_gradients(zip(g_gradient, self.gen_tv))

        return g_loss, g_mae_loss, g_pcp_loss, g_reg_loss,\
               psnr1, ssim1, mae1, I, L1, F1
               # psnr2, ssim2, mae2,\I, L1, \
               # L2, \
               # F1, \
               # F2

    @tf.function
    # def valid_step(self, image, label1, label2):
    def valid_step(self, image, label):
        # image = -image
        # label = -label
        logit = tf.tanh(self.G_ema(image))
        # fake1, fake2 = tf.split(logit, 2, -1)
        fake = logit
        # Loss
        g_mae_loss = self.gen_mae_loss(label, fake)
        #              + self.gen_mae_loss(label2, fake2)
        # g_mae_loss = self.gen_mae_loss(label1, fake1) \
        #              + self.gen_mae_loss(label2, fake2)
        # g_pcp_loss = self.gen_pcp_loss(label1, fake1) \
        #              + self.gen_pcp_loss(label2, fake2)
        g_pcp_loss = self.gen_pcp_loss(label, fake) \
            # + self.gen_pcp_loss(label2, fake2)
        g_reg_loss = self.gen_reg_loss()
        g_loss = g_mae_loss + g_pcp_loss + g_reg_loss

        # Metrics
        I = tf.clip_by_value(0.5 * (image + 1), 0, 1)
        F1 = tf.clip_by_value(0.5 * (fake + 1), 0, 1)
        # F2 = tf.clip_by_value(0.5 * (fake2 + 1), 0, 1)
        # L1 = tf.clip_by_value(0.5 * (label1 + 1), 0, 1)
        L1 = tf.clip_by_value(0.5 * (label + 1), 0, 1)
        # L2 = tf.clip_by_value(0.5 * (label2 + 1), 0, 1)
        psnr1 = tf.reduce_mean(tf.image.psnr(F1, L1, 1.0))
        ssim1 = tf.reduce_mean(tf.image.ssim(F1, L1, 1.0))
        mae1 = tf.reduce_mean(tf.abs(F1 - L1))
        # psnr2 = tf.reduce_mean(tf.image.psnr(F2, L2, 1.0))
        # ssim2 = tf.reduce_mean(tf.image.ssim(F2, L2, 1.0))
        # mae2 = tf.reduce_mean(tf.abs(F2 - L2))

        return g_loss, g_mae_loss, g_pcp_loss, g_reg_loss,\
               psnr1, ssim1, mae1, I, L1, F1
               # psnr2, ssim2, mae2,\
               # I, L1, L2, F1, F2

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
            # image, label1, label2 = next(self.train_iter)
            image, label = next(self.train_iter)
            #PET-MRI
            g_loss, g_mae_loss, g_pcp_loss, g_reg_loss, \
            psnr1, ssim1, mae1, \
            I, L1, F1 = self.train_step(image, label)
            # psnr2, ssim2, mae2, \


            # Compute moving average of network parameters
            self.moving_average(self.G, self.G_ema, beta=0.999)

            epoch = idx // self.step_per_epoch
            info = ("[Iter: %6d/%6d(%d)][Loss: %.3f]" \
                   "[MRI-PET: PSNR/SSIM/MAE: %.3f/%.3f/%.3f]" \
                   "[time: %s]"
                # "[NAC-AC: PSNR/SSIM/MAE: %.3f/%.3f/%.3f]" \
                    % (idx, self.iteration, epoch, g_loss,psnr1,
                      ssim1 * 100, mae1 * 100, datetime.now() - start_time))
                     # psnr2, ssim2 * 100, mae2 * 100,
            fprint(FLAGS.logdir + '/log.txt', info)

            if idx % self.step_per_epoch ==0:
                IMAGE, LABEL = next(self.valid_iter)
                OUTPUTS = self.valid_step(IMAGE, LABEL)

                with self.summary_1.as_default():
                    tf.summary.scalar('Lr/G', self.G_optimizer.lr(idx), epoch)
                    tf.summary.scalar('G/All_loss', g_loss, epoch)
                    tf.summary.scalar('G/Mae_loss', g_mae_loss, epoch)
                    tf.summary.scalar('G/Pcp_loss', g_pcp_loss, epoch)
                    tf.summary.scalar('G/Reg_loss', g_reg_loss, epoch)
                    tf.summary.scalar('toCT/PSNR', psnr1, epoch)
                    tf.summary.scalar('toCT/SSIM', ssim1, epoch)
                    tf.summary.scalar('toCT/MAE',   mae1, epoch)
                    # tf.summary.scalar('toAC/PSNR', psnr2, epoch)
                    # tf.summary.scalar('toAC/SSIM', ssim2, epoch)
                    # tf.summary.scalar('toAC/MAE',   mae2, epoch)
                    tf.summary.image('Input', I, epoch, 4)
                    tf.summary.image('Label1', L1, epoch, 4)
                    # tf.summary.image('Label2', L2, epoch, 4)
                    tf.summary.image('Logit1', F1, epoch, 4)
                    # tf.summary.image('Logit2', F2, epoch, 4)

                with self.summary_2.as_default():
                    tf.summary.scalar('G/All_loss', OUTPUTS[0], epoch)
                    tf.summary.scalar('G/Mae_loss', OUTPUTS[1], epoch)
                    tf.summary.scalar('G/Pcp_loss', OUTPUTS[2], epoch)
                    tf.summary.scalar('G/Reg_loss', OUTPUTS[3], epoch)
                    tf.summary.scalar('toPET/PSNR', OUTPUTS[4], epoch)
                    tf.summary.scalar('toPET/SSIM', OUTPUTS[5], epoch)
                    tf.summary.scalar('toPET/MAE',  OUTPUTS[6], epoch)
                    # tf.summary.scalar('toAC/PSNR', OUTPUTS[7], epoch)
                    # tf.summary.scalar('toAC/SSIM', OUTPUTS[8], epoch)
                    # tf.summary.scalar('toAC/MAE',  OUTPUTS[9], epoch)
                    tf.summary.image('Input',  OUTPUTS[7], epoch, 4)
                    tf.summary.image('Label1', OUTPUTS[8], epoch, 4)
                    # tf.summary.image('Label2', OUTPUTS[12], epoch, 4)
                    tf.summary.image('Logit1', OUTPUTS[9], epoch, 4)
                    # tf.summary.image('Logit2', OUTPUTS[14], epoch, 4)

                # save every self.save_freq
                if epoch % 10 == 0:
                    self.manager.save(checkpoint_number=idx)

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

    """============================ Testing and Results ==============================="""
    def test(self):
        self.save_dir = os.path.join('./outs', FLAGS.logdir[7:])
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)

        #Restore Ckpt
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')
        test_data = glob_image_from_folder(FLAGS.test_dir)
        psnr_1, ssim_1, mae_1, mape_1, nmae_1 = [], [], [], [], []
        # psnr_2, ssim_2, mae_2, mape_2, nmae_2 = [], [], [], [], []
        # psnr_3, ssim_3, mae_3, mape_3, nmae_3 = [], [], [], [], []

        for i, path in enumerate(test_data):
            print(path)
            # image, True1, True2 = read_jpg(path)
            # image, True1, True2 = read_png(path)
            image, label = read_image_without_augment(path)
            # Generation
            I = tf.concat([image],-1)
            logit = tf.tanh(self.G_ema(I[None])[0])
            logit = logit[:,:,:1]
            # Fake1, Fake2 = tf.split(logit, 2, -1)
            # fake1 = logit
            # normalize to [0, 1]
            image = tf.clip_by_value(image * 32767 + 32767, 0, 65535)
            true1 = -label
            true1 = tf.clip_by_value(true1 * 32767 + 32767, 0, 65535)
            fake1 = -logit
            fake1 = tf.clip_by_value(fake1 * 32767 + 32767, 0, 65535)


            # image = tf.clip_by_value(0.5 * image[:, :, :1] + 0.5, 0, 1)
            # # true1 = tf.clip_by_value(0.5 * True1[:,:,:1] + 0.5, 0, 1)
            # true1 = tf.clip_by_value(0.5 * label[:, :, :1] + 0.5, 0, 1)
            # # true2 = tf.clip_by_value(0.5 * True2[:,:,:1] + 0.5, 0, 1)
            # fake1 = tf.clip_by_value(0.5 * fake1[:, :, :1] + 0.5, 0, 1)
            # fake2 = tf.clip_by_value(0.5 * Fake2 + 0.5, 0, 1)
            comb = np.concatenate((image, true1, fake1), 1)
            save_name = self.save_dir + '/%04d.png' % i
            cv2.imwrite(save_name, comb.astype(np.uint16))

            save_name_1 = self.save_dir + '/logit'
            if not os.path.exists(save_name_1):
                os.makedirs(save_name_1)
            save_name_2 = self.save_dir + '/label'
            if not os.path.exists(save_name_2):
                os.makedirs(save_name_2)
            save_name_1 = self.save_dir + '/logit' + '/%04d.png' % i
            save_name_2 = self.save_dir + '/label' + '/%04d.png' % i
            cv2.imwrite(save_name_1, np.uint16(-fake1))
            cv2.imwrite(save_name_2, np.uint16(-true1))
            # cv2.imwrite(os.path.join(tmp_dir, path[-12:]), np.uint16(comb))
            # image_list = [[image], [true1, fake1]]
            # # image_list = [[image], [true1, fake1], [true2, fake2]]
            # draw_PIL_canvas(save_name, image_list, [1,2,2], True)

            # Metrics
            mae_1.append(tf.reduce_mean(tf.abs(true1 - fake1)).numpy())
            psnr_1.append(tf.reduce_mean(tf.image.psnr(true1, fake1, 65535)).numpy())
            ssim_1.append(tf.reduce_mean(tf.image.ssim(true1, fake1, 65535)).numpy())
            mape_1.append(tf.reduce_mean(tf.abs(tf.math.divide_no_nan(true1-fake1, true1))).numpy())
            nmae_1.append(tf.math.divide_no_nan(tf.reduce_sum(tf.abs(true1-fake1)), tf.reduce_sum(true1)).numpy())

            # mae_2.append(tf.reduce_mean(tf.abs(true2 - fake2)).numpy())
            # psnr_2.append(tf.reduce_mean(tf.image.psnr(true2, fake2, 1.0)).numpy())
            # ssim_2.append(tf.reduce_mean(tf.image.ssim(true2, fake2, 1.0)).numpy())
            # mape_2.append(tf.reduce_mean(tf.abs(tf.math.divide_no_nan(true2-fake2, true2))).numpy())
            # nmae_2.append(tf.math.divide_no_nan(tf.reduce_sum(tf.abs(true2-fake2)),tf.reduce_sum(true2)).numpy())
            #
            # mae_3.append(tf.reduce_mean(tf.abs(true2 - image)).numpy())
            # psnr_3.append(tf.reduce_mean(tf.image.psnr(true2, image, 1.0)).numpy())
            # ssim_3.append(tf.reduce_mean(tf.image.ssim(true2, image, 1.0)).numpy())
            # mape_3.append(tf.reduce_mean(tf.abs(tf.math.divide_no_nan(true2-image, true2))).numpy())
            # nmae_3.append(tf.math.divide_no_nan(tf.reduce_sum(tf.abs(true2-image)),tf.reduce_sum(true2)).numpy())


        save_csv(self.save_dir + '/metrics.csv',{
            'psnr_ct': psnr_1, 'ssim_ct': ssim_1, 'mae_ct': mae_1, 'mape_ct': mape_1,'nmae_ct': nmae_1})
            # 'psnr_ac': psnr_2, 'ssim_ac': ssim_2, 'mae_ac': mae_2, 'mape_ac': mape_2,'nmae_ac': nmae_2,
            # 'psnr_nac': psnr_3, 'ssim_nac': ssim_3, 'mae_nac': mae_3, 'mape_nac': mape_3,'nmae_nac': nmae_3

        save_file = self.save_dir + '/results.txt'
        fprint(save_file, str(datetime.now()))
        fprint(save_file, '==================================================')
        fprint(save_file, 'For MRI->PET: ')
        fprint(save_file, 'PSNR = %.4f +/- %.4f dB' % list_mean_std(psnr_1, 1))
        fprint(save_file, 'SSIM = %.4f +/- %.4f %%' % list_mean_std(ssim_1, 100))
        fprint(save_file, 'MAE = %.4f +/- %.4f HU' % list_mean_std(mae_1, 1))
        fprint(save_file, 'MAPE = %.4f +/- %.4f %%' % list_mean_std(mape_1, 100))
        fprint(save_file, 'NMAE = %.4f +/- %.4f %%' % list_mean_std(nmae_1, 100))
        fprint(save_file, '==================================================')
        # fprint(save_file, 'For NAC->AC: ')
        # fprint(save_file, 'PSNR = %.4f + %.4f dB' % list_mean_std(psnr_2, 1))
        # fprint(save_file, 'SSIM = %.4f + %.4f %%' % list_mean_std(ssim_2, 100))
        # fprint(save_file, 'MAE = %.4f + %.4f HU' % list_mean_std(mae_2, 1))
        # fprint(save_file, 'MAPE = %.4f + %.4f %%' % list_mean_std(mape_2, 100))
        # fprint(save_file, 'NMAE = %.4f + %.4f %%' % list_mean_std(nmae_2, 100))
        # fprint(save_file, '==================================================')
        # fprint(save_file, 'For |NAC-AC|: ')
        # fprint(save_file, 'PSNR = %.4f + %.4f dB' % list_mean_std(psnr_3, 1))
        # fprint(save_file, 'SSIM = %.4f + %.4f %%' % list_mean_std(ssim_3, 100))
        # fprint(save_file, 'MAE = %.4f + %.4f HU' % list_mean_std(mae_3, 1))
        # fprint(save_file, 'MAPE = %.4f + %.4f %%' % list_mean_std(mape_3, 100))
        # fprint(save_file, 'NMAE = %.4f + %.4f %%' % list_mean_std(nmae_3, 100))
        # fprint(save_file, '==================================================')

    def tmp(self):
        tmp_dir = os.path.join(self.save_dir, 'tmp')
        if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)
        self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
        print('Latest checkpoint restored!!')

        paths = glob_image_from_folder('./data/SET_5/axial')
        paths = paths[:573] # Subject 5

        psnr_1, ssim_1, mae_1, mape_1, nmae_1 = [], [], [], [], []
        psnr_2, ssim_2, mae_2, mape_2, nmae_2 = [], [], [], [], []
        psnr_3, ssim_3, mae_3, mape_3, nmae_3 = [], [], [], [], []

        for i, path in enumerate(paths):
            print(path[-12:])
            # image, True1, True2 = read_jpg(path)
            # image, True1, True2 = read_png(path)
            label, image = read_image_without_augment(path)
            # Generation
            logit = tf.tanh(self.G_ema(image[None])[0])
            Fake1, Fake2 = tf.split(logit, 2, -1)
            # normalize to [0, 1]
            image = tf.clip_by_value(0.5 * image[:,:,:1] + 0.5, 0, 1)
            # true1 = tf.clip_by_value(0.5 * True1[:,:,:1] + 0.5, 0, 1)
            true1 = tf.clip_by_value(0.5 * label[:,:,:1] + 0.5, 0, 1)
            # true2 = tf.clip_by_value(0.5 * True2[:,:,:1] + 0.5, 0, 1)
            fake1 = tf.clip_by_value(0.5 * Fake1[:,:,:1] + 0.5, 0, 1)
            # fake2 = tf.clip_by_value(0.5 * Fake2[:,:,:1] + 0.5, 0, 1)

            # image_list = [[image], [true1, fake1], [true2, fake2]]
            # draw_PIL_canvas(save_name, image_list, [1, 2, 2])

            # tf.keras.preprocessing.image.save_img(save_nac, image)
            # tf.keras.preprocessing.image.save_img(save_ac, true2)
            # tf.keras.preprocessing.image.save_img(save_sac, fake2)
            # tf.keras.preprocessing.image.save_img(save_ct, true1)
            # tf.keras.preprocessing.image.save_img(save_sct, fake1)


            # save_nac = tmp_dir +'/nac/' +  path[-12:]
            # save_ac = tmp_dir +'/ac/' +  path[-12:]
            # save_sac = tmp_dir +'/sac/' +  path[-12:]
            # save_ct = tmp_dir +'/ct/' +  path[-12:]
            # save_sct = tmp_dir +'/sct/' +  path[-12:]
            # if not os.path.exists(tmp_dir +'/nac'): os.makedirs(tmp_dir +'/nac')
            # if not os.path.exists(tmp_dir +'/ac'): os.makedirs(tmp_dir +'/ac')
            # if not os.path.exists(tmp_dir +'/sac'): os.makedirs(tmp_dir +'/sac')
            # if not os.path.exists(tmp_dir +'/ct'): os.makedirs(tmp_dir +'/ct')
            # if not os.path.exists(tmp_dir +'/sct'): os.makedirs(tmp_dir +'/sct')
            # cv2.imwrite(save_nac, np.uint16(image * 2000))
            # cv2.imwrite(save_ac, np.uint16(true2 * 15000))
            # cv2.imwrite(save_sac, np.uint16(fake2 * 15000))
            # cv2.imwrite(save_ct, np.uint16(true1 * 2000))
            # cv2.imwrite(save_sct, np.uint16(fake1 * 2000))

            mask = tf.ones_like(image) * 1000
            inputs= tf.concat((image*2000, true1*2000), 1)
            # inputs= tf.concat((image*2000, true1*2000, true2*15000), 1)
            # outputs = tf.concat((mask, fake1*2000, fake2*15000),1)
            outputs = tf.concat((mask, fake1*2000),1)
            comb = tf.concat((inputs, outputs), 0)


            # Metrics
        #     psnr_ct.append(tf.reduce_mean(tf.image.psnr(true1, fake1, 1.0)).numpy())
        #     ssim_ct.append(tf.reduce_mean(tf.image.ssim(true1, fake1, 1.0)).numpy())
        #     mae_ct.append(tf.reduce_mean(tf.abs(true1 - fake1)).numpy())
        #     psnr_ac.append(tf.reduce_mean(tf.image.psnr(true2, fake2, 1.0)).numpy())
        #     ssim_ac.append(tf.reduce_mean(tf.image.ssim(true2, fake2, 1.0)).numpy())
        #     mae_ac.append(tf.reduce_mean(tf.abs(true2 - fake2)).numpy())
        #     psnr_nac.append(tf.reduce_mean(tf.image.psnr(true2, image, 1.0)).numpy())
        #     ssim_nac.append(tf.reduce_mean(tf.image.ssim(true2, image, 1.0)).numpy())
        #     mae_nac.append(tf.reduce_mean(tf.abs(true2 - image)).numpy())
        #
        # save_csv(tmp_dir + '/metrics.csv',
        #          {'psnr_ct': psnr_ct, 'ssim_ct': ssim_ct, 'mae_ct': mae_ct,
        #           'psnr_ac': psnr_ac, 'ssim_ac': ssim_ac, 'mae_ac': mae_ac,
        #           'psnr_nac': psnr_nac, 'ssim_nac': ssim_nac, 'mae_nac': mae_nac})
        #
        # save_file = tmp_dir + '/results.txt'
        # fprint(save_file, str(datetime.now()))
        # fprint(save_file, '==================================================')
        # fprint(save_file, 'For NAC->CT: ')
        # fprint(save_file, 'PSNR = %.4f + %.4fdB' % list_mean_std(psnr_ct, 1))
        # fprint(save_file, 'SSIM = %.4f + %.4f%%' % list_mean_std(ssim_ct, 100))
        # fprint(save_file, 'MAE = %.4f + %.4f%%' % list_mean_std(mae_ct, 100))
        # fprint(save_file, '==================================================')
        # fprint(save_file, 'For NAC->AC: ')
        # fprint(save_file, 'PSNR = %.4f + %.4fdB' % list_mean_std(psnr_ac, 1))
        # fprint(save_file, 'SSIM = %.4f + %.4f%%' % list_mean_std(ssim_ac, 100))
        # fprint(save_file, 'MAE = %.4f + %.4f%%' % list_mean_std(mae_ac, 100))
        # fprint(save_file, '==================================================')
        # fprint(save_file, 'For |NAC-AC|: ')
        # fprint(save_file, 'PSNR = %.4f + %.4fdB' % list_mean_std(psnr_nac, 1))
        # fprint(save_file, 'SSIM = %.4f + %.4f%%' % list_mean_std(ssim_nac, 100))
        # fprint(save_file, 'MAE = %.4f + %.4f%%' % list_mean_std(mae_nac, 100))
        # fprint(save_file, '==================================================')

"""============================ Main ==============================="""

if __name__ == '__main__':
    run_on_gpu()
    check_logdir()
    cnn = MultiCNN()
    if FLAGS.phase == 'train':
        cnn.train()
    cnn.test()
    # cnn.tmp()




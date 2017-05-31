import cv2
import numpy as np
import NLDF
import os
import sys
import tensorflow as tf
import time
import vgg16


def load_img_list(dataset):

    if dataset == 'MSRA-B':
        path = 'dataset/MSRA-B/image'
    elif dataset == 'HKU-IS':
        path = 'dataset/HKU-IS/imgs'
    elif dataset == 'DUT-OMRON':
        path = 'dataset/DUT-OMRON/DUT-OMRON-image'
    elif dataset == 'PASCAL-S':
        path = 'dataset/PASCAL-S/pascal'
    elif dataset == 'SOD':
        path = 'dataset/BSDS300/imgs'
    elif dataset == 'ECSSD':
        path = 'dataset/ECSSD/images'

    imgs = os.listdir(path)

    return path, imgs


if __name__ == "__main__":

    model = NLDF.Model()
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    img_size = NLDF.img_size
    label_size = NLDF.label_size

    ckpt = tf.train.get_checkpoint_state('Model/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)

    datasets = ['MSRA-B', 'HKU-IS', 'DUT-OMRON',
                'PASCAL-S', 'ECSSD', 'SOD']

    if not os.path.exists('Result'):
        os.mkdir('Result')

    for dataset in datasets:
        path, imgs = load_img_list(dataset)

        save_dir = 'Result/' + dataset
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_dir = 'Result/' + dataset + '/NLDF'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for f_img in imgs:

            img = cv2.imread(os.path.join(path, f_img))
            img_name, ext = os.path.splitext(f_img)

            if img is not None:
                ori_img = img.copy()
                img_shape = img.shape
                img = cv2.resize(img, (img_size, img_size)) - vgg16.VGG_MEAN
                img = img.reshape((1, img_size, img_size, 3))

                start_time = time.time()
                result = sess.run(model.Prob,
                                  feed_dict={model.input_holder: img,
                                             model.keep_prob: 1})
                print("--- %s seconds ---" % (time.time() - start_time))

                result = np.reshape(result, (label_size, label_size, 2))
                result = result[:, :, 0]

                result = cv2.resize(np.squeeze(result), (img_shape[1], img_shape[0]))

                save_name = os.path.join(save_dir, img_name+'_NLDF.png')
                cv2.imwrite(save_name, (result*255).astype(np.uint8))

    sess.close()

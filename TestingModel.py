import cv2
import numpy as np
import NLDF as Model
import os, sys
import tensorflow as tf
import time
import vgg16

def load_training_list():
    with open('dataset/MSRA-B/train_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    files = []
    labels = []

    for line in lines:
        labels.append('dataset/MSRA-B/annotation/%s' % line)
        files.append('dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))

    return files, labels

def load_val_list():
    with open('dataset/MSRA-B/valid_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    files = []
    labels = []

    for line in lines:
        labels.append('dataset/MSRA-B/annotation/%s' % line)
        files.append('dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))

    return files, labels

def load_test_list():
    with open('dataset/MSRA-B/test_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    files = []
    labels = []

    for line in lines:
        labels.append('dataset/MSRA-B/annotation/%s' % line)
        files.append('dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))

    return files, labels


def Pascal_list():

    files = []
    labels = []

    for i in xrange(850):
        filename = 'dataset/Pascal-S/pascal/%d.jpg' % (i+1)
        files.append(filename)

        labelname = 'dataset/Pascal-S/mask/%d.png' % (i+1)
        labels.append(labelname)

    return files, labels

def HKUIS_list():

    files = []
    labels = []

    imgs = os.listdir('dataset/HKU-IS/imgs/')

    for img in imgs:
        filename = 'dataset/HKU-IS/imgs/%s' % (img)
        files.append(filename)

        label = img.replace('.jpg', '.png')
        labelname = 'dataset/HKU-IS/gt/%s' % (label)
        labels.append(labelname)

    return files, labels

def DUT_OMRON_list():

    files = []
    labels = []

    imgs = os.listdir('dataset/DUT-OMRON/DUT-OMRON-image/')

    for img in imgs:
        filename = 'dataset/DUT-OMRON/DUT-OMRON-image/%s' % (img)
        files.append(filename)

        label = img.replace('.jpg', '.png')
        labelname = 'dataset/DUT-OMRON/pixelwiseGT-new-PNG/%s' % (label)
        labels.append(labelname)

    return files, labels


if __name__ == "__main__":

    model = Model.DAC()
    model.build_model()


    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    img_size = Model.img_size
    label_size = Model.label_size

    ckpt = tf.train.get_checkpoint_state('Model/')
    saver = tf.train.Saver()
    saver.restore(sess, ckpt.model_checkpoint_path)


    train_list, label_list = load_test_list()
    #train_list, label_list = Pascal_list()
    #train_list, label_list = HKUIS_list()
    #train_list, label_list = DUT_OMRON_list()


    for f_img in train_list:
        #print f_img


        img = cv2.imread(f_img)
        ori_img = img.copy()
        img_shape= img.shape
        img = cv2.resize(img,(img_size,img_size)) - vgg16.VGG_MEAN
        img = img.reshape((1, img_size, img_size, 3))

        start_time = time.time()
        result = sess.run(model.Prob,
                          feed_dict={model.input_holder: img,
                                     model.keep_prob: 1})
        print("--- %s seconds ---" % (time.time() - start_time))

        result = np.reshape(result, (label_size, label_size,2))
        result = result[:,:,0]

        result = cv2.resize(np.squeeze(result),(img_shape[1],img_shape[0]))


        res = f_img.replace("/image/", "/result/NLDF/")    #MSRA-B
        #res = f_img.replace("/pascal/", "/V13_v2_C_v2/")   #Pascal
        #res = f_img.replace("/imgs/", "/V13_v2_C_v2/")      #HKUIS
        #res = f_img.replace("/DUT-OMRON-image/", "/V13_v2_C_v2/")  # DUT-OMRON

        res = res.replace(".jpg", ".png")
        cv2.imwrite(res, (result*255).astype(np.uint8))

    sess.close()

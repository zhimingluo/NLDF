import tensorflow as tf
import vgg16
import cv2
import numpy as np

img_size = 352
label_size = img_size / 2


class Model:
    def __init__(self):
        self.vgg = vgg16.Vgg16()

        self.input_holder = tf.placeholder(tf.float32, [1, img_size, img_size, 3])
        self.label_holder  = tf.placeholder(tf.float32, [label_size*label_size, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        self.weights = np.array([0.8, 0.2])

        self.sobel_fx, self.sobel_fy = self.sobel_filter()

        self.contour_th = 1.5
        self.contour_weight = 0.0001

    def build_model(self):

        #build the VGG-16 model
        vgg = self.vgg
        vgg.build(self.input_holder)

        fea_dim = 128

        #Global Feature and Global Score
        pool5_shape = vgg.pool5.get_shape()
        self.Fea_Global_1 = tf.nn.relu(self.Conv_2d(vgg.pool5, [5, 5, 512, fea_dim], 0.01, padding='VALID', name='Fea_Global_1'))
        self.Fea_Global_2 = tf.nn.relu(self.Conv_2d(self.Fea_Global_1, [5, 5, fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_Global_2'))
        self.Fea_Global = self.Conv_2d(self.Fea_Global_2, [3, 3, fea_dim, fea_dim], 0.01, padding='VALID', name='Fea_Global')

        #Local Score
        self.Fea_P5 = tf.nn.relu(self.Conv_2d(vgg.pool5, [3,3,512,fea_dim], 0.01, padding='SAME', name='Fea_P5'))
        self.Fea_P4 = tf.nn.relu(self.Conv_2d(vgg.pool4, [3,3,512,fea_dim], 0.01, padding='SAME', name='Fea_P4'))
        self.Fea_P3 = tf.nn.relu(self.Conv_2d(vgg.pool3, [3,3,256,fea_dim], 0.01, padding='SAME', name='Fea_P3'))
        self.Fea_P2 = tf.nn.relu(self.Conv_2d(vgg.pool2, [3,3,128,fea_dim], 0.01, padding='SAME', name='Fea_P2'))
        self.Fea_P1 = tf.nn.relu(self.Conv_2d(vgg.pool1, [3,3,64,fea_dim], 0.01, padding='SAME', name='Fea_P1'))

        self.Fea_P5_LC = self.Contrast_Layer(self.Fea_P5, 3)
        self.Fea_P4_LC = self.Contrast_Layer(self.Fea_P4, 3)
        self.Fea_P3_LC = self.Contrast_Layer(self.Fea_P3, 3)
        self.Fea_P2_LC = self.Contrast_Layer(self.Fea_P2, 3)
        self.Fea_P1_LC = self.Contrast_Layer(self.Fea_P1, 3)

        #Deconv Layer
        self.Fea_P5_Up = tf.nn.relu(self.Deconv_2d(tf.concat(3, [self.Fea_P5, self.Fea_P5_LC]), [1, 22, 22, fea_dim], 5, 2, name='Fea_P5_Deconv'))
        self.Fea_P4_Up = tf.nn.relu(self.Deconv_2d(tf.concat(3, [self.Fea_P4, self.Fea_P4_LC, self.Fea_P5_Up]), [1, 44, 44, fea_dim*2], 5, 2, name='Fea_P4_Deconv'))
        self.Fea_P3_Up = tf.nn.relu(self.Deconv_2d(tf.concat(3, [self.Fea_P3, self.Fea_P3_LC, self.Fea_P4_Up]), [1, 88, 88, fea_dim*3], 5, 2, name='Fea_P3_Deconv'))
        self.Fea_P2_Up = tf.nn.relu(self.Deconv_2d(tf.concat(3, [self.Fea_P2, self.Fea_P2_LC, self.Fea_P3_Up]), [1, 176, 176, fea_dim*4], 5, 2, name='Fea_P2_Deconv'))

        self.Local_Fea = self.Conv_2d(tf.concat(3, [self.Fea_P1, self.Fea_P1_LC, self.Fea_P2_Up]), [1,1,fea_dim*6, fea_dim*5], 0.01, padding='VALID', name='Local_Fea')
        self.Local_Score = self.Conv_2d(self.Local_Fea, [1,1,fea_dim*5,2], 0.01, padding='VALID', name='Local_Score')

        self.Global_Score = self.Conv_2d(self.Fea_Global, [1, 1, fea_dim, 2], 0.01, padding='VALID', name='Global_Score')

        self.Score = self.Local_Score + self.Global_Score
        self.Score = tf.reshape(self.Score, [-1,2])

        self.Prob = tf.nn.softmax(self.Score)

        #Get the contour term
        self.Prob_C = tf.reshape(self.Prob, [1, 176, 176, 2])
        self.Prob_Grad = tf.tanh(self.im_gradient(self.Prob_C))
        self.Prob_Grad = tf.tanh(tf.reduce_sum(self.im_gradient(self.Prob_C), reduction_indices=3, keep_dims=True))

        self.label_C = tf.reshape(self.label_holder, [1, 176, 176, 2])
        self.label_Grad = tf.cast(tf.greater(self.im_gradient(self.label_C),self.contour_th), tf.float32)
        self.label_Grad = tf.cast(tf.greater(tf.reduce_sum(self.im_gradient(self.label_C), reduction_indices=3, keep_dims=True),self.contour_th), tf.float32)

        self.C_IoU_LOSS = self.Loss_IoU(self.Prob_Grad, self.label_Grad)
        self.Contour_Loss = self.Loss_Contour(self.Prob_Grad, self.label_Grad)

        #Loss Function
        self.Loss_Mean = self.C_IoU_LOSS + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.Score, self.label_holder))

        self.correct_prediction = tf.equal(tf.argmax(self.Score,1), tf.argmax(self.label_holder,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def Conv_2d(self, input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding= padding)

            b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            conv = tf.nn.bias_add(conv, b)

            return conv

    def Deconv_2d(self, input_, output_shape,
                  k_s=3, st_s=2, stddev=0.01, padding='SAME',name="deconv2d"):
        with tf.variable_scope(name):
            W = tf.get_variable('W',
                                shape=[k_s, k_s, output_shape[3], input_.get_shape()[3]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

            deconv = tf.nn.conv2d_transpose(input_, W, output_shape=output_shape,
                                            strides=[1, st_s, st_s, 1], padding=padding)

            b = tf.get_variable('b', [output_shape[3]], initializer=tf.constant_initializer(0.0))
            deconv = tf.nn.bias_add(deconv, b)

        return deconv

    def Contrast_Layer(self, input_, k_s=3):
        h_s = k_s / 2
        return tf.sub(input_, tf.nn.avg_pool(tf.pad(input_, [[0,0], [h_s,h_s], [h_s,h_s], [0,0]], 'SYMMETRIC'),
                                             ksize=[1, k_s, k_s, 1], strides=[1, 1, 1, 1], padding='VALID'))

    def sobel_filter(self):
        fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
        fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)

        fx = np.stack((fx, fx), axis=2)
        fy = np.stack((fy, fy), axis=2)

        fx = np.reshape(fx, (3, 3, 2, 1))
        fy = np.reshape(fy, (3, 3, 2, 1))

        #tf_fx = tf.constant(fx)
        #tf_fy = tf.constant(fy)

        tf_fx = tf.Variable(tf.constant(fx))
        tf_fy = tf.Variable(tf.constant(fy))


        return tf_fx, tf_fy

    def im_gradient(self, im):
        gx = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'), self.sobel_fx, [1, 1, 1, 1], padding='VALID')
        gy = tf.nn.depthwise_conv2d(tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC'), self.sobel_fy, [1, 1, 1, 1], padding='VALID')
        return tf.sqrt(tf.add(tf.square(gx), tf.square(gy)))

    def Loss_IoU(self, pred, gt):
        inter = tf.reduce_sum(tf.mul(pred, gt))
        union = tf.add(tf.reduce_sum(tf.square(pred)), tf.reduce_sum(tf.square(gt)))

        if inter == 0:
            return 0
        else:
            return 1 - (2*(inter+1)/(union + 1))

    def Loss_Contour(self, pred, gt):
        return tf.reduce_mean(-gt*tf.log(pred+0.00001) - (1-gt)*tf.log(1-pred+0.00001))

    def L2(self, tensor, wd=0.0005):
        return tf.mul(tf.nn.l2_loss(tensor), wd, name='L2-Loss')

if __name__ == "__main__":

    img = cv2.imread("dataset/MSRA-B/0_1_1339.jpg")

    h, w = img.shape[0:2]
    img = cv2.resize(img,(img_size,img_size)) - vgg16.VGG_MEAN
    img = img.reshape((1, img_size, img_size, 3))

    label = cv2.imread("dataset/MSRA-B/0_1_1339.png")[:, :, 0]
    label = cv2.resize(label,(label_size,label_size))
    label = label.astype(np.float32) / 255
    label = np.stack((label, 1-label), axis=2)
    label = np.reshape(label, [-1,2])

    sess = tf.Session()

    model = DAC()
    model.build_model()

    #train_op = tf.train.AdamOptimizer(0.0001)


    max_grad_norm = 1
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.Loss_Mean, tvars), max_grad_norm)
    opt = tf.train.AdamOptimizer(0.00001)
    #opt = tf.train.MomentumOptimizer(0.001, momentum=0.9)
    optimizer = opt.apply_gradients(zip(grads, tvars))

    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(model.Loss_Mean)

    sess.run(tf.initialize_all_variables())

    for i in xrange(200):
        _, loss, C_IoU_LOSS = sess.run([optimizer, model.Loss_Mean, model.C_IoU_LOSS],
                            feed_dict={model.input_holder: img,
                                       model.label_holder: label,
                                       model.keep_prob: 0.5})
        print loss, C_IoU_LOSS

    result,boundary, gt_boundary = sess.run([tf.nn.softmax(model.Score), model.Prob_Grad, model.label_Grad],
                                            feed_dict={model.input_holder: img,
                                                       model.label_holder: label,
                                                       model.keep_prob: 1})

    result = np.reshape(result[:,0], (label_size,label_size))
    result = cv2.resize(result, (w, h))

    boundary = np.squeeze(boundary)
    boundary = cv2.resize(boundary, (w, h))

    gt_boundary = np.squeeze(gt_boundary)
    gt_boundary = cv2.resize(gt_boundary, (w, h))

    cv2.imshow('result', np.uint8(result*255))
    cv2.imshow('boundary', np.uint8(boundary*255))
    cv2.imshow('boundary_gt', np.uint8(gt_boundary*255))

    cv2.waitKey()

    cv2.imwrite('boundary.png', np.uint8(boundary*255))
    cv2.imwrite('boundary_gt.png', np.uint8(gt_boundary*255))

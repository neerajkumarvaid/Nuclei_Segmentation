import sys
caffe_root = '/home/sanuj/Projects/BTP/caffe'
sys.path.insert(0, caffe_root + '/python')
import caffe
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy

w = 33
p = (w-1)/2

caffe.set_mode_gpu()

# net = caffe.Net(caffe_root+'/examples/nuclei/train_cifar/deploy.prototxt',
#                 caffe_root+'/examples/nuclei/train_cifar/92_accu_use_cifar_model/use_cifar_2_iter_20000.caffemodel',
#                 caffe.TEST)

# net = caffe.Net(caffe_root+'/examples/nuclei/train_cifar/deploy1.prototxt',
#                 caffe_root+'/examples/nuclei/train_cifar/use_cifar_6_iter_40000.caffemodel',
#                 caffe.TEST)

net = caffe.Net(caffe_root+'/examples/nuclei/multi_class_nuclei/use_multi_class/deploy.prototxt',
                caffe_root+'/examples/nuclei/multi_class_nuclei/use_multi_class/use_multi_class_nuclei_big_1_iter_19186.caffemodel',
                caffe.TEST)

# net = caffe.Net(caffe_root+'/examples/nuclei/deploy.prototxt',
#                 caffe_root+'/examples/nuclei/nuclei_quick_relative_iter_100000.caffemodel',
#                 caffe.TEST)
# 63_LLM_YR4_cropped.jpg
# 81_LLM_YR4.jpg
im = mpimg.imread('/home/sanuj/Projects/BTP/data/63_LLM_YR4_cropped.jpg')
height, width, channel = im.shape
im = np.pad(im, ((p,p),(p,p),(0,0)), 'constant', constant_values=255)

label_im = np.zeros((height, width, channel))   # label image
prob_im = np.zeros((height, width, channel))    # probability image
aux_im = np.zeros((height, width))              # auxiliary image
# num_zeros = 0
b = 0
prob_x, prob_y = 0, 0
for i in range(p, p + height):
    for j in range(p, p + width):
        if not (im[i,j] >= [220, 220, 220]).all():
            aux_im[i-p, j-p] = 1
            for c in xrange(3):
                net.blobs['data'].data[b, c, :, :] = im[i-p:i+p+1, j-p:j+p+1, c]
            b += 1
            if b == 500:
                out = net.forward()
                print i, j

                for k in out['prob']:
                    while not aux_im[prob_x, prob_y]:
                        prob_y += 1
                        if prob_y == width:
                            prob_y = 0
                            prob_x += 1

                    if aux_im[prob_x, prob_y]:
                        prob_im[prob_x, prob_y, :] = k[1]
                        prob_y += 1
                        if prob_y == width:
                            prob_y = 0
                            prob_x += 1

                b = 0
            # if out['prob'][0][1] > 0.5:
            #     label_im[i-p,j-p,0]=label_im[i-p,j-p,1]=label_im[i-p,j-p,2] = 255*

#plots probability mask
prob_im = 255 * prob_im
prob_im = prob_im.astype(int)
scipy.misc.imsave('prob_im.jpg', prob_im)

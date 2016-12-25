import numpy as np
import matplotlib.image as mpimg
import scipy.misc
from sklearn.utils import shuffle
import scipy.io as sio

root = '/home/sanuj/Projects/'

file_name = root + 'nuclei-net/data/training-data/20x/2px/PrognosisTMABlock3_A_2_1_H&E.jpg'
mask_name = root + 'nuclei-net/data/training-data/20x/2px/tm_PrognosisTMABlock3_A_2_1_H&E.png'
# file_name = root + 'nuclei-net/data/training-data/63_LLM_YR4.jpg'
# mask_name = root + 'nuclei-net/data/training-data/tm_63_LLM_YR4.png'
w = 51
p = (w-1)/2
num = 18000
stride = 8

im = mpimg.imread(file_name).astype('uint8')
mask = (mpimg.imread(mask_name)*2).astype('uint8')
# mask = mask/np.amax(mask)
height, width, channel = im.shape
image = []
for i in range(channel):
    image.append(im[:,:,i])
image = np.array(image)
# image = np.pad(image, ((0,0),(p,p),(p,p)), 'constant', constant_values=255)
# mask = np.pad(mask, p, 'constant') # default constant_values=0

x = []

for i in range(0, np.amax(mask)+1):
	x.append([])

h = 0;
done = False
# for i in range(p, p+height):
for i in range(p, height-p):
# for i in range(325, height-p):
	if done:
		break
	# for j in range(p, p+width):
	for j in range(p, width-p, stride):
	# for j in range(55, width-p):
		total_len = 0
		for k in x:
			total_len += len(k)
		if total_len == num:
			done = True
			break
		if not (image[0,i,j] >= 220 and image[1,i,j] >= 220 and image[2,i,j] >= 220):
		    temp_x = image[:, i-p:i+p+1, j-p:j+p+1]
		    if len(x[mask[i,j]]) < num/3:
				print 'Label: ' + str(mask[i,j]) + '   len: ' + str(total_len) + '   i: ' + str(i) + '   j: ' + str(j)
				x[mask[i,j]].append(temp_x.flatten())
				if mask[i,j]:
					for k in range(0, 3):
						if len(x[mask[i,j]]) < num/3:
							temp_x = np.rot90(temp_x)
							print 'Label: ' + str(mask[i,j]) + '   len: ' + str(total_len) + '   i: ' + str(i) + '   j: ' + str(j)
							x[mask[i,j]].append(temp_x.flatten())
						else:
							break

data = np.concatenate(x)
label = []
for i in range(num):
	label.append(int(i/(num/3)))

label = np.array(label).astype('uint8')

data, label = shuffle(data, label, random_state=0)

dict = {'data': data, 'label': label}

sio.savemat(root + 'PrognosisTMABlock3_A_2_1_H&E_' + str(w) + '_2px_18000.mat', dict)
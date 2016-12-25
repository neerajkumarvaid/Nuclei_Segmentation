local matio = require 'matio'

num = 120000
c = 3
w = 51

ten = {}
ten['data'] = torch.Tensor(num, c, w, w):byte()
ten['label'] = torch.Tensor(num):byte()

t = matio.load('PrognosisTMABlock1_E_4_5_H&E_51_2px_51000.mat')
t['data'] = torch.reshape(t['data'], 51000, c, w, w)
t['label'] = torch.reshape(t['label'], 51000)

ten['data'][{ {1, 51000}, {}, {}, {} }] = t['data']
ten['label'][{ {1, 51000} }] = t['label']

t = matio.load('PrognosisTMABlock3_A_2_1_H&E_1_51_2px_51000.mat')
t['data'] = torch.reshape(t['data'], 51000, c, w, w)
t['label'] = torch.reshape(t['label'], 51000)

ten['data'][{ {51001, 102000}, {}, {}, {} }] = t['data']
ten['label'][{ {51001, 102000} }] = t['label']

t = matio.load('PrognosisTMABlock3_A_2_1_H&E_51_2px_18000.mat')
t['data'] = torch.reshape(t['data'], 18000, c, w, w)
t['label'] = torch.reshape(t['label'], 18000)

ten['data'][{ {102001, 120000}, {}, {}, {} }] = t['data']
ten['label'][{ {102001, 120000} }] = t['label']

torch.save('/home/sanuj/Projects/20x_2px_train_120000.t7', ten)
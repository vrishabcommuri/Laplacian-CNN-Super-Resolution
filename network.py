"""
##########################################

python script for upscaling an input image

##########################################
Citations

The Net and _Conv_Block classes are adapted from the LapSRN paper:

paper: http://vllab.ucmerced.edu/wlai24/LapSRN/papers/cvpr17_LapSRN.pdf
github: https://github.com/twtygqyy/pytorch-LapSRN

file interface adapted from:

github: https://github.com/BUPTLdy/Pytorch-LapSRN/blob/master/test.py

"""




import torch 
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, utils
from collections import OrderedDict
from PIL import Image
from torchvision.utils import save_image
import scipy.misc
import scipy.io as sio
import time
import math


class _Conv_Block(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cov_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self, x):
        out = self.cov_block(x)
        return out

    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_input = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.convt_I1 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F1 = self.make_layer(_Conv_Block)
        
        self.convt_I2 = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.convt_R2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convt_F2 = self.make_layer(_Conv_Block)
        
    def make_layer(self, block):
        layers = []
        layers.append(block())
        return nn.Sequential(*layers)
    
    def forward(self, x):    
        out = self.relu(self.conv_input(x))
        
        convt_F1 = self.convt_F1(out)
        convt_I1 = self.convt_I1(x)
        convt_R1 = self.convt_R1(convt_F1)
        HR_2x = convt_I1 + convt_R1
        
        convt_F2 = self.convt_F2(convt_F1)
        convt_I2 = self.convt_I2(HR_2x)
        convt_R2 = self.convt_R2(convt_F2)
        HR_4x = convt_I2 + convt_R2
       
        return HR_2x, HR_4x




def upscale(name, data, factor):
	img = Image.open(str(name))
	img = img.convert('L')

	pix_gt = img.resize((factor,factor), Image.NEAREST)
	pix_gt.save(str(data.timename)+'down.jpg')
	pix_up = pix_gt.resize((4*factor,4*factor), Image.BICUBIC)
	pix_up.save(str(data.timename)+'up_bicubic.jpg')
	pix_gt_arr = np.array(pix_gt)
	pix_down = img.resize((factor,factor), Image.BICUBIC)
	pix_down_arr = np.array(pix_down)
	sio.savemat('./np_matrix_input.mat', {'down':pix_down_arr})
	cnn = Net()
	cnn.load_state_dict(torch.load('params.pkl'))

	low_res = sio.loadmat("./np_matrix_input.mat")['down']

	low_res = low_res.astype(float) 
	_input = low_res/255.0
	variable_input = Variable(torch.from_numpy(_input).float()).view(1, -1, _input.shape[0], _input.shape[1]).cuda()
	variable_input.size()
	cnn = cnn.cuda()
	start_time = time.time()
	HR_2x, HR_4x = cnn(variable_input)
	elapsed_time = time.time() - start_time
	HR_4x = HR_4x.cpu()

	hi_res = HR_4x.data.squeeze_(0)
	hi_res.size()

	print(elapsed_time)

	postprocess = transforms.Compose([
    transforms.ToPILImage()
	])

	y = postprocess(hi_res)
	y.save(str(data.timename)+"your_file.jpg")
	y = y.resize((1400,1400), Image.BICUBIC)
	return y

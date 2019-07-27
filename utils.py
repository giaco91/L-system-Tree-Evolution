import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import sys
from copy import deepcopy
import math
import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle

import matplotlib.pyplot as plt


# np.random.seed(7)



def create_image(i, j):
	image = Image.new("RGB", (i, j), "white")
	return image

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def sigmoid(s,bias=0):
	return 1/(1+np.exp(-s))+bias

def resize_to_height_ref(image,n_height):
    w,h=image.size
    return image.resize((n_height,round(n_height*h/w)),Image.ANTIALIAS)

def resize(image,W,H):
	return image.resize((W,H),Image.ANTIALIAS)

def reflect_y_axis(im):
	return ImageOps.mirror(im).rotate(180,expand=True)

def intersection_of_two_lines(p1,d1,p2,d2):
	#we define the lines: l1=p1+t[0]*d1 and l2=p2+t[1]*d2
	A=np.zeros((2,2))
	A[:,0]=d1
	A[:,1]=-d2
	det=np.linalg.det(A)
	if abs(det)<1e-10 or False:
		return np.array([None,None])
	else:
		A_inv=np.linalg.inv(A) 
		t=np.dot(A_inv,p2-p1)
		#s=p1+t[0]*d1
		return t

def rotation(alpha,v):
	c=np.cos(alpha)
	s=np.sin(alpha)
	R=np.array([[c,-s],[s,c]])
	return np.dot(R,v)

def get_angle(v1,v2):
	alpha=np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
	if math.isnan(alpha):
		alpha=0#I tont know how else to solve it. if v1 is almost equal to v2 something strange happens and alpha becomes none
	return alpha

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pack_sequences(data,with_alpha=False):
	#the data is a list of sequences. Every sequence is it self a list of data points.
    #the data must be preprocessed in order to use nn.utils.rnn.pack_padded_sequence
    #sequence size:(batchsize,max_seq_length,data_dim)
    n_sequences=len(data)
    sequence_lengths=np.zeros(n_sequences).astype(int)
    for i in range(n_sequences):
    	sequence_lengths[i]=int(len(data[i]))
    sorted_idx=np.argsort(sequence_lengths)
    sorted_sequence_lengths=[]
    data_batch=torch.zeros(n_sequences,sequence_lengths[sorted_idx[-1]],5)
    for i in range(n_sequences):
    	if with_alpha:
    		data_idx=np.asarray([0,1,2,4,5,6])
    	else:
    		data_idx=np.asarray([0,1,2,4,5])
    	data_batch[i,0:sequence_lengths[sorted_idx[-1-i]],:]=torch.from_numpy(np.asarray(data[sorted_idx[-1-i]])[:,data_idx])
    	sorted_sequence_lengths.append(sequence_lengths[sorted_idx[-1-i]])
    packed_data_batch = nn.utils.rnn.pack_padded_sequence(data_batch, sorted_sequence_lengths, batch_first=True)
    return packed_data_batch


def unpack_sequences(data_packed):
    #the inverse of pack sequences
    data, seq_lengths= nn.utils.rnn.pad_packed_sequence(data_packed, batch_first=True)
    return data,seq_lengths

def get_shifted_background(background_im,scaled_ds):
	fade_step=10
	W,H=background_im.size
	new_background_im=background_im.copy()
	px=background_im.load()
	new_px=new_background_im.load()
	for w in range(W):
		for h in range(H):
			if px[w,h]==(230,200,0):
				px[w,h]=(255,255,255)
			elif px[w,h]!=(255,255,255):
				px[w,h]=(min(px[w,h][0]+fade_step,255),min(px[w,h][1]+fade_step,255),min(px[w,h][2]+fade_step,255))
	for w in range(W):
		for h in range(H):
			shifted_w_idx=round(w+scaled_ds[0])
			shifted_h_idx=round(h+scaled_ds[1])
			if 0<=shifted_w_idx<=W-1 and 0<=shifted_h_idx<=H-1:
				new_px[w,h]=px[shifted_w_idx,shifted_h_idx]
	return new_background_im

def get_max_size(line_segments):
	min_x=0
	max_x=0
	min_y=0
	max_y=0
	print('calculate tree dimensions ...')
	for i in range(len(line_segments)):
		if line_segments[i][0][0]<min_x:
			min_x=line_segments[i][0][0]
		if line_segments[i][0][0]>max_x:
			max_x=line_segments[i][0][0]
		if line_segments[i][0][1]<min_y:
			min_y=line_segments[i][0][1]
		if line_segments[i][0][1]>max_y:
			max_y=line_segments[i][0][1]
	if line_segments[-1][1][0]<min_x:
		min_x=line_segments[i][1][0]
	if line_segments[-1][1][0]>max_x:
		max_x=line_segments[i][1][0]
	if line_segments[-1][1][1]<min_y:
		min_y=line_segments[i][1][1]
	if line_segments[-1][1][1]>max_y:
		max_y=line_segments[i][1][1]

	x_mean=(max_x+min_x)/2
	return max(max_x-min_x,max_y-min_y),x_mean




def draw_tree(line_segments,im_size=500,width=1):
	im=create_image(im_size,im_size)
	max_size,x_mean=get_max_size(line_segments)
	scale=0.9*im_size/max_size
	W,H=im.size
	W_2=int(W/2)
	H_2=int(H/2)
	bias=np.array([W_2-scale*x_mean,0])
	draw = ImageDraw.Draw(im)
	print('draw tree...')
	for i in range(len(line_segments)):
		draw.line([tuple(bias+scale*line_segments[i][0]),tuple(bias+scale*line_segments[i][1])],fill=(0,0,0),width=width)
	return reflect_y_axis(im)
















import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import sys
from copy import deepcopy
import math
import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw, ImageFont
import PIL
import numpy as np
import pickle
from copy import deepcopy
import matplotlib.pyplot as plt


# np.random.seed(7)



def create_image(i, j,color=(255,255,255)):
	image = Image.new("RGB", (i, j), color=color)
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


def get_max_size(line_segments):
	min_x=0
	max_x=0
	min_y=0
	max_y=0
	for i in range(len(line_segments)):
		if line_segments[i][0][0]<min_x:
			min_x=line_segments[i][0][0]
		if line_segments[i][0][0]>max_x:
			max_x=line_segments[i][0][0]
		if line_segments[i][0][1]<min_y:
			min_y=line_segments[i][0][1]
		if line_segments[i][0][1]>max_y:
			max_y=line_segments[i][0][1]
	if len(line_segments)>0:
		if line_segments[-1][1][0]<min_x:
			min_x=line_segments[i][1][0]
		if line_segments[-1][1][0]>max_x:
			max_x=line_segments[i][1][0]
		if line_segments[-1][1][1]<min_y:
			min_y=line_segments[i][1][1]
		if line_segments[-1][1][1]>max_y:
			max_y=line_segments[i][1][1]

	x_mean=(max_x+min_x)/2
	return max(max_x-min_x,max_y-min_y),x_mean,min_y


def draw_tree(line_segments,im_size=500,width=1):
	im=create_image(im_size,im_size)
	if len(line_segments)>0:
		max_size,x_mean,y_min=get_max_size(line_segments)
		if max_size==0:
			print('something went wrong, because max_size is zero')
			max_size=1
		scale=0.9*im_size/max_size
		W,H=im.size
		W_2=int(W/2)
		H_2=int(H/2)
		bias=np.array([W_2-scale*x_mean,-scale*y_min])
		draw = ImageDraw.Draw(im)
		#print('draw tree...')
		for i in range(len(line_segments)):
			coord1=bias+scale*line_segments[i][0]
			coord2=bias+scale*line_segments[i][1]
			depth=line_segments[i][2]
			# d=np.sqrt(np.sum(np.power(coord1-coord2,2)))
			# width=max(1,int(d/5))
			width=max(1,15-2*depth)
			color=(max(0,100-int(5*depth)),min(255,10+int(depth*10)),10)
			draw.line([tuple(coord1),tuple(coord2)],fill=color,width=width)
	return reflect_y_axis(im)

def draw_trees(trees,im_size=500,width=1):
	#trees is a list of line_segments for each tree
	im=create_image(im_size,im_size)
	draw = ImageDraw.Draw(im)
	n_trees=len(trees)
	n_rows=int(np.ceil(np.sqrt(n_trees)))
	W,H=im.size
	W_n_rows=int(W/n_rows)
	H_n_rows=int(H/n_rows)
	W_n_2=int(W_n_rows/2)
	H_n_2=int(H_n_rows/2)
	i=0
	#print('draw tree...')
	for wr in range(n_rows):
		for hr in range(n_rows):
			if i<=len(trees)-1:
				if len(trees[i])>0:
					max_size,x_mean,y_min=get_max_size(trees[i])
					scale=0.9*W_n_rows/max_size
					bias=np.array([W_n_2-scale*x_mean,-scale*y_min])
					for j in range(len(trees[i])):
						BIAS=np.array([wr*W_n_rows,hr*H_n_rows])
						draw.line([tuple(BIAS+bias+scale*trees[i][j][0]),tuple(BIAS+bias+scale*trees[i][j][1])],fill=(0,50,0),width=width)
			i+=1
	return reflect_y_axis(im)

def draw_sequence_of_branches(branches_list,text_list,im_size=1000,draw_leafs=True,draw_stress=True, max_stress_only=True):
	image_list=[]
	global_max_size=0
	global_x_min=0
	global_x_max=0
	global_y_min=0
	global_y_max=0
	global_x_mean=0
	global_y_min=0
	W_2=int(im_size/2)
	H_2=W_2
	for b in branches_list:
		min_x,max_x,min_y,max_y=from_branches_get_max_size(b,get_extreme_only=True)
		if global_x_min>min_x:
			global_x_min=min_x
		if global_x_max<max_x:
			global_x_max=max_x
		if global_y_min>min_y:
			global_y_min=min_y
		if global_y_max<max_y:
			global_y_max=max_y
	global_max_size=max(global_x_max-global_x_min,global_y_max-global_y_min)
	scale=0.7*im_size/(global_max_size+1e-10)
	if global_max_size/2<global_x_max:
		global_x_mean=global_x_max-global_max_size/2
	elif global_max_size/2<-global_x_min:
		global_x_mean=global_max_size/2-global_x_max
	bias=np.array([0.4*im_size+W_2-scale*global_x_mean,0.05*im_size-scale*global_y_min])
	for i in range(len(branches_list)):
		image_list.append(draw_branches(branches_list[i],im_size=im_size,text=text_list[i],draw_leafs=draw_leafs,draw_stress=draw_stress, max_stress_only=max_stress_only,scale=scale,bias=bias))
	return image_list

def draw_branches(branches,im_size=1000,text=None,draw_leafs=True,draw_stress=True, max_stress_only=True,scale=None,bias=None):
	#text must be alist of strings
	max_stress=get_max_stress(branches)
	im=create_image(int(1.4*im_size),im_size)
	draw = ImageDraw.Draw(im)
	if len(branches)>0:
		dx,dy,x_mean,y_min,_=from_branches_get_max_size(branches)
		max_size=max(dx,dy)
		if max_size==0:
			print('something went wrong, because max_size is zero')
			max_size=1
		#W,H=im.size
		W_2=int(im_size/2)
		H_2=W_2
		if scale is None or bias is None:
			scale=0.7*im_size/max_size
			bias=np.array([0.4*im_size+W_2-scale*x_mean,0.05*im_size-scale*y_min])
		#print('draw tree...')
		for i in range(len(branches)):
			if draw_stress:
				if max_stress_only:
					if abs(branches[i].stress())==max_stress:
						if branches[i].stress()>0:
							draw.line([tuple(bias+scale*branches[i].sc),tuple(bias+scale*branches[i].ec)],fill=(int(255*branches[i].stress()/max_stress),0,0),width=max(1,int(round(scale*branches[i].cs))))
						else:
							draw.line([tuple(bias+scale*branches[i].sc),tuple(bias+scale*branches[i].ec)],fill=(0,0,int(-255*branches[i].stress()/max_stress)),width=max(1,int(round(scale*branches[i].cs))))
					else:
						draw.line([tuple(bias+scale*branches[i].sc),tuple(bias+scale*branches[i].ec)],fill=(0,0,0),width=max(1,int(round(scale*branches[i].cs))))
				else:
					if branches[i].stress()>0:
						draw.line([tuple(bias+scale*branches[i].sc),tuple(bias+scale*branches[i].ec)],fill=(int(255*branches[i].stress()/max_stress),0,0),width=max(1,int(round(scale*branches[i].cs))))
					else:
						draw.line([tuple(bias+scale*branches[i].sc),tuple(bias+scale*branches[i].ec)],fill=(0,0,int(-255*branches[i].stress()/max_stress)),width=max(1,int(round(scale*branches[i].cs))))
			else:
				draw.line([tuple(bias+scale*branches[i].sc),tuple(bias+scale*branches[i].ec)],fill=(0,0,0),width=max(1,int(round(scale*branches[i].cs))))
			if draw_leafs:
				if branches[i].leaf_radius is not None:
					box_coord=tuple(bias+scale*(branches[i].ec-branches[i].leaf_radius))+tuple(bias+scale*(branches[i].ec+branches[i].leaf_radius))
					draw.ellipse(box_coord, fill=(100,255,100), outline=(0,255,0))
	im_reflected= reflect_y_axis(im)
	draw = ImageDraw.Draw(im_reflected)
	if text is not None:
		dy=0
		for i in range(len(text)):
			font = ImageFont.truetype("arial.ttf", int(40*im_size/1000))
			draw.text((int(0.01*im_size),int(0.01*im_size+dy)), text[i], font=font, fill=(0,0,0))
			dy+=1.1*font.getsize(text[i])[1]

	return im_reflected



def draw_trees_from_branches(trees,im_size=1000):
	#trees is a list of branches for each tree
	im=create_image(im_size,im_size)
	draw = ImageDraw.Draw(im)
	n_trees=len(trees)
	n_rows=int(np.ceil(np.sqrt(n_trees)))
	W,H=im.size
	W_n_rows=int(W/n_rows)
	H_n_rows=int(H/n_rows)
	W_n_2=int(W_n_rows/2)
	H_n_2=int(H_n_rows/2)
	MAX_SIZE=0
	for k in range(n_trees):
		dx,dy,_,_,_=from_branches_get_max_size(trees[k])
		max_size=max(dx,dy)
		if max_size>MAX_SIZE:
			MAX_SIZE=max_size
	i=0
	for wr in range(n_rows):
		for hr in range(n_rows):
			if i<=len(trees)-1:
				if len(trees[i])>0:
					dx,dy,x_mean,y_min,_=from_branches_get_max_size(trees[i])
					max_size=max(dx,dy)
					if max_size==0:
						print('something went wrong, because max_size is zero')
						max_size=1
					scale=0.9*W_n_rows/MAX_SIZE
					bias=np.array([W_n_2-scale*x_mean,-scale*y_min])
					for j in range(len(trees[i])):
						BIAS=np.array([wr*W_n_rows,hr*H_n_rows])
						draw.line([tuple(BIAS+bias+scale*trees[i][j].sc),tuple(BIAS+bias+scale*trees[i][j].ec)],fill=(0,0,0),width=max(1,int(round(scale*trees[i][j].cs))))
			i+=1
	return reflect_y_axis(im)


def from_branches_get_max_size(branches,get_extreme_only=False):
	min_x=0
	max_x=0
	min_y=0
	max_y=0
	for i in range(len(branches)):
		if branches[i].sc[0]<min_x:
			min_x=branches[i].sc[0]
		if branches[i].sc[0]>max_x:
			max_x=branches[i].sc[0]
		if branches[i].sc[1]<min_y:
			min_y=branches[i].sc[1]
		if branches[i].sc[1]>max_y:
			max_y=branches[i].sc[1]
		if branches[i].ec[0]<min_x:
			min_x=branches[i].ec[0]
		if branches[i].ec[0]>max_x:
			max_x=branches[i].ec[0]
		if branches[i].ec[1]<min_y:
			min_y=branches[i].ec[1]
		if branches[i].ec[1]>max_y:
			max_y=branches[i].ec[1]
	if get_extreme_only:
		return min_x,max_x,min_y,max_y

	x_mean=(max_x+min_x)/2
	y_mean=(max_y+min_y)/2

	return max_x-min_x,max_y-min_y,x_mean,min_y,y_mean

def get_leaf_projection(branches,angle):
	e_angle=rotation(angle,np.array([1,0]))
	projected_scalars=[]
	average_leaf_height=0
	leaf_radius=0
	for b in branches:
		if b.leaf_radius is not None:
			projected_scalars.append(np.dot(b.ec,e_angle))
			average_leaf_height
			average_leaf_height+=b.ec[1]
			if leaf_radius==0:
				leaf_radius=b.leaf_radius
	n_leafs=len(projected_scalars)
	if n_leafs>0:
		average_leaf_height/=n_leafs
	projected_scalars.sort()
	projected_area=0
	b=-1e10
	for s in projected_scalars:
		if b<s-leaf_radius:
			projected_area+=2*leaf_radius
		else:
			projected_area+=s-b+leaf_radius
		b=s+leaf_radius
	return projected_area,n_leafs,average_leaf_height



def get_max_stress(branches):
	if len(branches)>0:
		max_stress=-1e10
		for b in branches:
			if abs(b.stress())>max_stress:
				max_stress=abs(b.stress())
		return max_stress
	else:
		return 0

def get_average_stress(branches):
	if len(branches)>0:
		stress=0
		for b in branches:
			stress+=abs(b.stress())
		return stress/len(branches)
	else:
		return 0

def get_euclidean_stress(branches):
	if len(branches)>0:
		stress=0
		for b in branches:
			stress+=b.stress()**2
		return np.sqrt(stress)/len(branches)
	else:
		return 0

def get_total_mass(branches):
	if len(branches)>0:
		mass=0
		for b in branches:
			mass+=b.mass
		return mass
	else:
		return 0

def get_morph_score(branches,TI,w,dl,verbose=False,ground=True):
	dx,dy,x_mean,y_min,y_mean=from_branches_get_max_size(branches)
	area_score=dx*dy
	stability_score_x=-(x_mean)**4
	stability_score_y=-(y_mean-3)**4
	stability_score=stability_score_x+stability_score_y
	if verbose:
		print('area score: ',str(area_score))
		print('stability_score x: ',str(stability_score_x))
		print('stability_score y: ',str(stability_score_y))
	if ground:
		ground_score=-10*min(y_min,0)**2
		if verbose:
			print('ground score: ',str(ground_score))
	return area_score + ground_score +stability_score

def get_score(branches,TI,w,dl,verbose=False):
	#calculates the score for a given tree described by the given list of branches
	dx,dy,x_mean,y_min,mean_y=from_branches_get_max_size(branches)
	#average_stress=get_average_stress(branches)
	max_stress=get_max_stress(branches)
	left_branches=TI.render(w,dl,starting_root=np.array([0,0]),starting_angle=np.pi/2+np.pi/4)
	right_branches=TI.render(w,dl,starting_root=np.array([0,0]),starting_angle=np.pi/2-np.pi/4)
	left_wind_branches=TI.render(w,dl,starting_root=np.array([0,0]),starting_angle=0,side_force=0.2)
	right_wind_branches=TI.render(w,dl,starting_root=np.array([0,0]),starting_angle=0,side_force=-0.2)
	_,_,_,y_min_left,_=from_branches_get_max_size(left_branches)
	_,_,_,y_min_right,_=from_branches_get_max_size(right_branches)
	#average_stress_left=get_average_stress(left_branches)
	max_stress_left=get_max_stress(left_branches)
	max_stress_left_wind=get_max_stress(left_wind_branches)
	#average_stress_right=get_average_stress(right_branches)
	max_stress_right=get_max_stress(right_branches)
	max_stress_right_wind=get_max_stress(right_wind_branches)
	stress_exponent=0.3
	#wind_stress=0*average_stress_left+0*average_stress_right+0*max_stress_left+0*max_stress_right
	wind_stress=np.exp(stress_exponent*min(10,max_stress_left))+np.exp(stress_exponent*min(10,max_stress_right))
	side_wind_stress=np.exp(stress_exponent*min(10,max_stress_left_wind))+np.exp(stress_exponent*min(10,max_stress_right_wind))
	wind_stress+=side_wind_stress
	mass=get_total_mass(branches)
	#a=-10*np.exp(-0.1*dx*dy)#maximize area of tree up to saturation
	#b=-0.1*(dy-3)**2#tree hight is optimal at 3
	#c=-abs(x_mean)**2#symmetry
	#h=0.15*len(branches)**0.5
	ground=(10*min(y_min,0))**2+0.1*((10*min(y_min_left,0))**2+(10*min(y_min_right,0))**2)#no branches with negative y-coordinates
	stress=2*(np.exp(stress_exponent*min(10,max_stress))+wind_stress)
	
	tot_p_a=0
	N=10
	phi=np.pi/3
	for i in range(N):
		projected_area,n_leafs,average_leaf_height=get_leaf_projection(branches,phi*((2*i/N-1)))
		tot_p_a+=projected_area
	average_sunlight=30*tot_p_a/N
	_,_,average_leaf_height=get_leaf_projection(branches,0)
	#leaf_cost=n_leafs*(TI.leaf_radius**2)
	leaf_cost=n_leafs*np.exp(min(10,6*TI.leaf_radius))#
	energy_loss=-mass-leaf_cost
	#height=1*max(0,mean_y)**0.5
	height=10*(1-np.exp(-min(10,max(0,average_leaf_height))))
	score=-ground-stress+energy_loss+average_sunlight+height+n_leafs**0.5
	if verbose:
		print('side_wind_stress: '+str(side_wind_stress))
		print('mass loss: '+str(-mass))
		print('leaf radius: '+str(TI.leaf_radius))
		print('ground '+str(-ground))
		print('stress: '+str(-stress))
		print('energy_loss: '+str(energy_loss))
		print('average_sunlight: '+str(average_sunlight))
		print('height: '+str(height))
	return score

def get_reduced_moment(cumulative_moment):
	#cumulative_momentthe accumulated moments with respect to the origin in the form of nested lists [[l1*f1,f1],[l2*f2,f2],...,]
	#we want to calculate the equivalent reduced form L*F=l1*f1+l2*f2+...
	F=0
	lm=0
	L=0
	for i in range(len(cumulative_moment)):
		lm+=cumulative_moment[i][0]
		F+=cumulative_moment[i][1]
	if len(cumulative_moment)>0:
		L=lm/F
	return L,F

def make_gif(pil_images,save_path='tree_evolution.gif',duration=500):
	print('create gif ...')
	pil_images[0].save(save_path,
	              save_all=True,
	              append_images=pil_images[1:],
	              duration=duration,
	              loop=0)

def list_to_string(l,a=0,b=1):
	#a,b, the interval
	string=''
	for e in l:
		string+=str(e)[a:b]
	return string

def get_rotated_stress(branches,angle):
	rot_branches=[]
	for b in branches:
		rot_b=deepcopy(b)
		rot_sc=rotation(angle,rot_b.sc)
		rot_ec=rotation(angle,rot_b.ec)
		rot_b.update_position(rot_sc,rot_ec)
		rot_branches.append(rot_b)

def selection(ls_list,ti_list,w_list,dl_list,trees,n_sel=1):
	l=len(ti_list)
	scores=np.zeros(l)
	for i in range(l):
		w=w_list[i]
		dl=dl_list[i]
		b=trees[i]
		#scores[i]=get_score(b,ti_list[i],w,dl)
		scores[i]=get_morph_score(b,ti_list[i],w,dl)
	sorted_idx=np.argsort(scores)
	selected_ls=[]
	selected_ti=[]
	for j in range(n_sel):
		selected_ls.append(ls_list[sorted_idx[-1-j]])
		selected_ti.append(ti_list[sorted_idx[-1-j]])
	return selected_ls, selected_ti,scores[sorted_idx[-1]],trees[sorted_idx[-1]]




# def selection(ls_list,ti_list,n_sel=1,depth_specific=False):
# 	l=len(ls_list)
# 	scores=np.zeros(l)
# 	trees=[]
# 	for i in range(l):
# 		w,dl=ls_list[i].evolution('X',ti_list[i].depth,depth_specific=depth_specific)
# 		b=ti_list[i].render(w,dl)
# 		scores[i]=get_score(b,ti_list[i],w,dl)
# 		trees.append(b)
# 	sorted_idx=np.argsort(scores)
# 	selected_ls=[]
# 	selected_ti=[]
# 	for j in range(n_sel):
# 		selected_ls.append(ls_list[sorted_idx[-1-j]])
# 		selected_ti.append(ti_list[sorted_idx[-1-j]])
# 	return selected_ls, selected_ti,scores[sorted_idx[-1]],trees[sorted_idx[-1]]

def mutation(selected_ls,selected_ti,n_mut=10,letter=['X'],leaf_specific=False,max_character=15,max_depth=4):
	#generates the next generation of mutates from the survivors:
	#takes a list of l-systems selected_ls and the corresponding list of tree intrepreter selected_ti
	#and the number of deduced mutants (f)or the next generation) n_mut.
	#For the mode leaf_specific we need to take care of the case where the depth mutates. We
	#then need to move the leaf specific rule to its new depth.

	if len(selected_ls)!= len(selected_ti):
		raise ValueError('the number of selected l-systems is not equal the number of selected tree interpreter')
	L=len(selected_ti)
	mut_ls=[]
	mut_ti=[]
	if n_mut<=L:
		print('Warning: more selected than mutated -> no selection!')
		mut_ls+=selected_ls[:n_mut]
		mut_ti+=selected_ti[:n_mut]
	else:
		mut_ls+=selected_ls[:L]
		mut_ti+=selected_ti[:L]
		i=L
		while i<=n_mut-1:
			idx=i%L
			mut_i_ls=deepcopy(selected_ls[idx])
			mut_i_ti=deepcopy(selected_ti[idx])
			mut_i_ti.angle_and_size_mutation()
			old_depth=mut_i_ti.depth
			mut_i_ti.depth_mutation(p=0.1,max_depth=max_depth,min_depth=2)
			new_depth=mut_i_ti.depth
			if leaf_specific:
				mut_i_ls.general_mutation('X',p=0.02,immune={'[',']'},words={'[X]','[F]'},max_character=max_character)
				if old_depth!=new_depth:
					mut_i_ls.add_rule('X'+str(new_depth), mut_i_ls.rule['X'+str(old_depth)])
					mut_i_ls.rule.pop('X'+str(old_depth))
				mut_i_ls.general_mutation('X'+str(new_depth),p=0.02,immune={'[',']'},words={'[X]','[F]'},max_character=max_character)
			else:	
				for l in letter:
					old_str=mut_i_ls.rule['X']
					mut_i_ls.general_mutation(l,p=0.05,immune={'[',']'},words={'[X]','[F]'},max_character=max_character)
					if mut_i_ls.rule['X'].count('X')>=5:
						# print('too many X: '+mut_i_ls.rule['X']+'. Changed to: '+old_str)
						mut_i_ls.rule['X']=old_str
			mut_ls.append(mut_i_ls)
			mut_ti.append(mut_i_ti)
			i+=1
	return mut_ls,mut_ti





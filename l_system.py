import sys
from copy import deepcopy
import random

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps, ImageDraw
import PIL
import numpy as np
import pickle

import matplotlib.pyplot as plt

from utils import *


class L_system():
	def __init__(self,rule={},vocabulary=None):
		self.vocabulary=vocabulary#the vocabulary does not need to explicitely specified since it is implied by the axiom and the rule
		self.rule=rule#a dictionary of maps from an element of the vocabulary to the space of words constructed by the vocabulary.

	def update_vocabulary(self, V):
		#V is a set of vocabs.
		if self.vocabulary is not None:
			self.vocabulary.update(V)
		else:
			self.vocabulary=V

	def remove_vocab(self,v):
		#v is a single vocab
		if self.vocabulary is None:
			print('The vocab can not be removed because the vocabulary is not defined')
		elif v not in self.vocabulary:
			print('The vocab can not be removed because the vocabulary does not contain it')
		else:
			self.covabulary.discard(v)

	def point_mutation(self,v,p,immune={}):
		#--point mutation: mutates every letter of the word of the rule v with probability probability p
		#immune is a set of letter which are immune to point_mutation
		if self.vocabulary is None:
			raise ValueError('You must introduce a vocabulary in order to use the point mutation function.')
		if v not in self.vocabulary  and len(v)<=1:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		for i in range(len(self.rule[v])):
			if np.random.rand()<p:
				sub=random.sample(self.vocabulary, 1)[0]
				if len(immune.intersection({sub,self.rule[v][i]})) ==0:
					#print('point mutation')
					var=self.rule[v][:i]+sub
					if i+1<=len(self.rule[v])-1:
						var+=self.rule[v][i+1:]
					self.rule[v]=var

	def perm_mutation(self,v,p):
		if v not in self.vocabulary and len(v)<=1:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		for i in range(len(self.rule[v])-1):
			if np.random.rand()<p:
				#print('permutation mutation')
				var=''
				if i-1>=0:
					var+=self.rule[v][:i]
				var+=self.rule[v][i+1]+self.rule[v][i]
				if i+2<=len(self.rule[v])-1:
					var+=self.rule[v][i+2:]
				self.rule[v]=var

	def sample_word(self,extension={},immune={},length=1):
		extended_set=self.vocabulary.union(extension)
		extended_set.difference(immune)
		sampled_word=''
		i=0
		while i<=length-1:
			add=random.sample(extended_set, 1)[0]
			if add not in immune:
				sampled_word+=add
				i+=1
		return sampled_word



	def add_mutation(self,v,p,words={},immune={},max_character=12):
		#words allows to extend the vocabulary by words from which can also be selected
		if v not in self.vocabulary and len(v)<=1:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		extended_set=self.vocabulary.union(words)
		new_w=self.rule[v]
		j=0
		for i in range(len(self.rule[v])-1):
			if np.random.rand()<p:
				add=random.sample(extended_set, 1)[0]
				if add not in immune:
					#print('add mutation')
					new_w=new_w[:i+j]+add+new_w[i+j:]
					j+=1
		if len(new_w)<=max_character:
			self.rule[v]=new_w

	def loss_mutation(self,v,p,immune={},min_character=2):
		if v not in self.vocabulary and len(v)<=1:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		new_w=self.rule[v]
		j=0
		for i in range(len(self.rule[v])-1):
			if np.random.rand()<p and self.rule[v][i] not in immune:
				#print('loss mutation')
				new_w=new_w[:i-j]+new_w[i+1-j:]
				j+=1
		if len(new_w)>=min_character:
			self.rule[v]=new_w


	def add_rule(self, v, w,verbose=False):
		#v: an element from the vocabulary (single string element)
		#w: a word over the vocabulary
		#note that add_rule(v,v) is the identity rule
		if self.vocabulary is not None:
			for i in range(len(w)):
				if w[i] not in self.vocabulary:
					raise ValueError('The rule is not allowed since it maps outside of the specified vocabulary. Update the vocabulary if needed.')
		if verbose:
			if v in self.rule:
				print('rule for '+v+' replaced')
		self.rule[v]=w

	def iter_step(self,w,depth_list,depth,depth_specific=False):
		#w is a word over the vocabulary set (if it has symbols outside of the vocabulary set, the identity map is assumed)
		next_w=''
		next_depth_list=[]
		for i in range(len(w)):
			current_w=w[i]
			if depth_specific:
				current_w+=str(depth)
			if current_w in self.rule:
				sub=self.rule[current_w]
				next_w+=sub
				for j in range(len(sub)):
					next_depth_list.append(depth)
			elif depth_specific and w[i] in self.rule:
					sub=self.rule[w[i]]
					next_w+=sub
					for j in range(len(sub)):
						next_depth_list.append(depth_list[i])
					#print('used for default value for rule: '+str(w[i]))
			else:
				next_w+=w[i]
				next_depth_list.append(depth_list[i])
		return next_w,next_depth_list

	def general_mutation(self,v,p=0.01,immune={},words={},max_character=15):
		self.loss_mutation(v,p,immune=immune)
		self.point_mutation(v,min(1,2*p),immune=immune)
		self.perm_mutation(v,p)
		self.add_mutation(v,p,immune=immune,words=words,max_character=max_character)

	def evolution(self,w,n_iter,depth_specific=False):
		#w is a word (also called axiom in this context) over the vocabulary set (if it has symbols outside of the vocabulary set, the identity map is assumed)
		#n_iter, the amount of iterations
		#print('calculate DNA...')
		depth_list=[]
		for i in range(len(w)):
			depth_list.append(0)#the root has same depth than the first iteration
		for n in range(n_iter):
			if depth_specific:
				w,depth_list=self.iter_step(w,depth_list,depth=n,depth_specific=True)
			else:
				w,depth_list=self.iter_step(w,depth_list,depth=n)
		return w,depth_list

class Tree_interpreter():
	#this interpreter turns a string with the standard vocabulary {'X','F','-','+','[',']'} into a set of line segments by the 
	#rule of turtle walk
	def __init__(self,plus_angle=np.pi/8,minus_angle=-np.pi/8,randomize_angle=False):
		self.p_angle=plus_angle
		self.m_angle=minus_angle
		self.randomize=randomize_angle
		self.ex=np.array([1,0])

	def angle_mutation(self):
		# self.p_angle*=np.random.rand()/2+0.75
		# self.m_angle*=np.random.rand()/2+0.75
		self.p_angle=min(np.pi,self.p_angle*(np.random.rand()/2+0.75))
		self.m_angle=max(-np.pi,self.m_angle*(np.random.rand()/2+0.75))

	def render(self,w):
		#w must be a word over the standard alphabet
		#returns a list of line segment objects represented by the start coordinates (x_s,y_s) and end coordinates (x_e,y_e)
		#the starting point is (0,0) and starting direction is (0,1)
		#print('calculate line segments ...')
		starting_root=np.array([0,0])
		starting_angle=np.pi/2
		line_segments,_=self.rendering_recursion(starting_root,starting_angle,w)
		return line_segments

	def rendering_recursion(self,root,root_angle,w,depth=1):
		current_angle=root_angle
		current_root=root
		line_segments=[]
		i=0
		length_w=len(w)
		while i<=length_w-1:
			if w[i]=='[':
				depth+=1
				additional_line_segments, n=self.rendering_recursion(current_root,current_angle,w[i+1:],depth)
				line_segments+=additional_line_segments
				i+=n
			elif w[i]==']':
				return line_segments,i+1
			elif w[i]=='F':
				next_root=current_root+rotation(current_angle,self.ex)
				line_segments.append([current_root,next_root,depth])
				current_root=next_root
			elif w[i]=='+':
				if self.randomize:
					current_angle+=self.p_angle*(np.random.rand()/2+0.75)
				else:
					current_angle+=self.p_angle
			elif w[i]=='-':
				if self.randomize:
					current_angle+=self.m_angle*(np.random.rand()/2+0.75)
				else:
					current_angle+=self.m_angle
			i+=1
		return line_segments,length_w



class Depth_specific_tree_interpreter():
	#this interpreter turns a string with the standard vocabulary {'X','F','-','+','[',']'} into a set of line segments by the 
	#rule of turtle walk. However, in this depth specific case the length of the move "F" and the angles "+/-" can depend on the iteration depth
	def __init__(self,depth=2,plus_angles=[np.pi/8,np.pi/8],minus_angles=[-np.pi/8,np.pi/8],lengths=[1,1],cross_sections=[0.1,0.1],leaf_radius=0.2,leaf_density=0):
		if depth!=len(plus_angles) or depth!=len(minus_angles) or depth!=len(lengths) or depth!=len(cross_sections):
			raise ValueError('the depth must be in accordance with the number of the depth parameters.')
		self.depth=depth
		self.p_angles=plus_angles
		self.m_angles=minus_angles
		self.lengths=lengths
		self.cs=cross_sections
		self.ex=np.array([1,0])
		self.leaf_radius=leaf_radius
		self.leaf_mass=leaf_density*leaf_radius**2


	def angle_and_size_mutation(self):
		self.leaf_radius*=np.random.rand()/2+0.75
		for i in range(self.depth):
			self.p_angles[i]=min(np.pi,self.p_angles[i]*(np.random.rand()/2+0.75))
			self.m_angles[i]=max(-np.pi,self.m_angles[i]*(np.random.rand()/2+0.75))
			self.lengths[i]*=np.random.rand()/2+0.75
			self.cs[i]=max(0.02,self.cs[i]*(np.random.rand()/2+0.75))

	def depth_mutation(self,p=0.05,max_depth=8,min_depth=2):
		p/=2
		r=np.random.rand()
		if r<p:
			if self.depth<=max_depth-1:
				self.modify_depth(self.depth+1)
		elif r>(1-p):
			if self.depth>=min_depth+1:
				self.modify_depth(self.depth-1)

	def modify_depth(self,new_depth):
		if new_depth<1:
			raise ValueError('depth can not be smaller than one!')
		delta=new_depth-self.depth
		self.depth=new_depth
		if delta>0:
			for i in range(delta):
				self.p_angles.append(self.p_angles[-1])
				self.m_angles.append(self.m_angles[-1])
				self.lengths.append(self.lengths[-1])
				self.cs.append(self.cs[-1])
		else:
			for i in range(delta):
				del self.p_angles[-1]
				del self.m_angles[-1]
				del self.lengths[-1]
				del self.cs[-1]


	def render(self,w,depth_list,starting_root=np.array([0,0]),starting_angle=np.pi/2,side_force=None):
		#w must be a word over the standard alphabet
		#returns a list of branch objects 
		#the starting point is (0,0) and starting direction is (0,1)
		#print('calculate branch objects ...')
		branches,_,_,_=self.rendering_recursion(starting_root,starting_angle,w,depth_list,side_force=side_force)
		return branches

	def rendering_recursion(self,root,root_angle,w,depth_list,side_force=None):
		#side_force points to east. Hence, a negative side_force points to west
		current_angle=root_angle
		current_root=root
		branches=[]
		side_branches=[]
		i=0
		cumulative_moment_x=[]#the accumulated moments with respect to the origin in the form of nested lists [[l1*m1,m1],[l2*m2,m2],...,]
		cumulative_moment_y=[]
		length_w=len(w)
		last_rec=False
		while i<=length_w-1:
			if w[i]=='[':
				additional_branches, n,cm_x,cm_y=self.rendering_recursion(current_root,current_angle,w[i+1:],depth_list[i+1:])
				i+=n
				L,M=get_reduced_moment(cm_x)
				if side_force is not None:
					L_y,F_y=get_reduced_moment(cm_y)
				cumulative_moment_x+=cm_x
				for j in range(len(branches)):
					# print(branches[j].moment)
					branches[j].add_moment(M*(L-branches[j].sc[0]))
					if side_force is not None:
						branches[j].add_moment(F_y*(L_y-branches[j].sc[1]))
					# print(branches[j].moment)
				if len(additional_branches)>0:
					last_rec=True
					side_branches+=additional_branches
			elif w[i]==']':
				if len(branches)>0 and not last_rec:
					#add leaf
					branches[-1].add_leaf(self.leaf_radius,self.leaf_mass)
					cumulative_moment_x.append([branches[-1].ec[0]*self.leaf_mass,self.leaf_mass])
					if side_force is not None:
						cumulative_moment_y.append([branches[-1].ec[1]*(self.leaf_radius**2)*side_force,(self.leaf_radius**2)*side_force])
					for j in range(len(branches)):
						# print(branches[-1-j].moment)
						branches[-1-j].add_moment(branches[-1].leaf_mass*(branches[-1].ec[0]-branches[-1-j].sc[0]))
						if side_force is not None:
							branches[-1-j].add_moment((self.leaf_radius**2)*side_force*(branches[-1].ec[1]-branches[-1-j].sc[1]))
						# print(branches[-1-j].moment)
				return branches+side_branches,i+1,cumulative_moment_x,cumulative_moment_y
			elif w[i]=='F':
				last_rec=False
				next_root=current_root+self.lengths[depth_list[i]]*rotation(current_angle,self.ex)
				branch=Branch(self.cs[depth_list[i]],current_root,next_root)
				if side_force is not None:
					branch.add_moment(side_force*(branch.ec[1]-branch.sc[1])*np.sqrt(branch.cs)*(branch.center_of_mass_coordinates[1]-branch.sc[1]))
				for j in range(len(branches)):
					# print(branches[j].moment)
					branches[j].add_moment(branch.mass*(branch.center_of_mass_coordinates[0]-branches[j].sc[0]))
					if side_force is not None:
						branches[j].add_moment(side_force*(branch.ec[1]-branch.sc[1])*np.sqrt(branch.cs)*(branch.center_of_mass_coordinates[1]-branches[j].sc[1]))
					# print(branches[j].moment)
				cumulative_moment_x.append([branch.center_of_mass_coordinates[0]*branch.mass,branch.mass])
				if side_force is not None:
					force=side_force*(branch.ec[1]-branch.sc[1])*np.sqrt(branch.cs)
					cumulative_moment_y.append([branch.center_of_mass_coordinates[1]*force,force])
				branches.append(branch)
				current_root=next_root
			elif w[i]=='+':
				current_angle+=self.p_angles[depth_list[i]]
			elif w[i]=='-':
				current_angle+=self.m_angles[depth_list[i]]
			i+=1
		if len(branches)>0 and not last_rec:
			#add leaf
			branches[-1].add_leaf(self.leaf_radius,self.leaf_mass)
			for j in range(len(branches)):
				# print(branches[-1-j].moment)
				branches[-1-j].add_moment(branches[-1].leaf_mass*(branches[-1].ec[0]-branches[-1-j].sc[0]))
				if side_force is not None:
					branches[-1-j].add_moment((self.leaf_radius**2)*side_force*(branches[-1].ec[1]-branches[-1-j].sc[1]))
				# print(branches[-1-j].moment)
		return branches+side_branches,length_w,cumulative_moment_x,cumulative_moment_y

class Branch():
	#this interpreter turns a string with the standard vocabulary {'X','F','-','+','[',']'} into a set of line segments by the 
	#rule of turtle walk. However, in this depth specific case the length of the move "F" and the angles "+/-" can depend on the iteration depth
	def __init__(self,cross_section,start_coordinates,end_coordinates,leaf_radius=None,leaf_mass=None):
		#start and end coordinates are numpy arrays
		self.sc=start_coordinates
		self.ec=end_coordinates
		self.center_of_mass_coordinates=(self.sc+self.ec)/2
		self.l=np.linalg.norm(end_coordinates-start_coordinates)
		self.cs=cross_section
		self.mass=self.l*self.cs**2
		self.max_moment=self.cs**2#normalized by e-module and geometric constant
		self.moment=self.mass*(self.center_of_mass_coordinates[0]-self.sc[0])#the moment acting at the start position of the branche normalized by gravity constant
		self.leaf_radius=leaf_radius
		self.leaf_mass=leaf_mass

	def add_moment(self,delta_moment):
		self.moment+=delta_moment

	def stress(self):
		return self.moment/self.max_moment

	def update_position(self,sc,ec):
		self.sc=sc
		self.ec=ec
		self.center_of_mass_coordinates=(self.sc+self.ec)/2
		self.l=np.linalg.norm(end_coordinates-start_coordinates)
		self.mass=self.l*self.cs**2
		self.moment=self.mass*(self.center_of_mass_coordinates[0]-self.sc[0])#any additional moment is lost

	def update_cross_section(self,cs):
		self.cs=cs
		self.mass=self.l*self.cs**2
		self.max_moment=self.cs**2
		self.moment=self.mass*(self.center_of_mass_coordinates[0]-self.sc[0])#any additional moment is lost

	def add_leaf(self,leaf_radius,leaf_mass):
		self.leaf_radius=leaf_radius
		self.leaf_mass=leaf_mass










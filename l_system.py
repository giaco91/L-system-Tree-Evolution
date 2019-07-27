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


#–---CAR-----
class L_system():
	def __init__(self,rule={},vocabulary=None):
		self.vocabulary=vocabulary#the vocabulary does not need to explicitely specified since it is implied by the axiom and the rule
		self.rule=rule#a dictionary of maps from an element of the vocabulary to the space of words under the vocabulary.

	def update_vocabulary(self, V):
		#V is a set of vocabs.
		if self.vocabulary is not None:
			self.vocabulary.update(V)
		else:
			self.vocabulary=V

	def remove_vocab(self,v):
		#v is a single vocab
		if self.vocabulary is None:
			print('The vocab can not removed because the vocabulary is not defined')
		elif v not in self.vocabulary:
			print('The vocab can not removed because the vocabulary does not contain it')
		else:
			self.covabulary.discard(v)

	def point_mutation(self,v,p,immune={}):
		#--point mutation: mutates every letter of the word of the rule v with probability probability p
		#immune is a set of letter which are immune to point_mutation
		if self.vocabulary is None:
			raise ValueError('You must introduce a vocabulary in order to use the point mutation function.')
		if v not in self.vocabulary:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		for i in range(len(self.rule[v])):
			if np.random.rand()<p:
				sub=random.sample(self.vocabulary, 1)[0]
				if len(immune.intersection({sub,self.rule[v][i]})) ==0:
					print('point mutation')
					var=self.rule[v][:i]+sub
					if i+1<=len(self.rule[v])-1:
						var+=self.rule[v][i+1:]
					self.rule[v]=var

	def perm_mutation(self,v,p):
		if v not in self.vocabulary:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		for i in range(len(self.rule[v])-1):
			if np.random.rand()<p:
				print('permutation mutation')
				var=''
				if i-1>=0:
					var+=self.rule[v][:i]
				var+=self.rule[v][i+1]+self.rule[v][i]
				if i+2<=len(self.rule[v])-1:
					var+=self.rule[v][i+2:]
				print(self.rule[v])
				print(var)
				self.rule[v]=var

	def add_mutation(self,v,p,words={},immune={}):
		#words allows to extend the vocabulary by words from which can also be selected
		if v not in self.vocabulary:
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
					print('add mutation')
					new_w=new_w[:i+j]+add+new_w[i+j:]
					j+=1
		self.rule[v]=new_w

	def loss_mutation(self,v,p,immune={}):
		if v not in self.vocabulary:
			raise ValueError('The leter '+v+' is not in the vocabulary')
		elif v not in self.rule:
			print('note that there is no explicite rule for the letter '+v+'. We introduce it as the identity rule.')
			self.add_rule(v,v)
		new_w=self.rule[v]
		j=0
		for i in range(len(self.rule[v])-1):
			if np.random.rand()<p and self.rule[v][i] not in immune:
				print('loss mutation')
				new_w=new_w[:i-j]+new_w[i+1-j:]
				j+=1
		self.rule[v]=new_w


	def add_rule(self, v, w,verbose=False):
		#v: an element from the vocabulary (single string element)
		#w: a word over the vocabulary
		#note that add_rule(v,v) is the identity rule
		if self.vocabulary is not None:
			for i in range(len(w)):
				if w[i] not in self.vocabulary:
					raise ValueError('The rule is not allowed since it maps outside of the specified vocabulary. Update the vocabulary if needed.')
		if len(v)>=2:
			raise ValueError('the input of the rule must be a single element from the vocabulary, but given word of length '+str(len(v)))
		else:
			if verbose:
				if v in self.rule:
					print('rule for '+v+' replaced')
			self.rule[v]=w

	def iter_step(self,w):
		#w is a word over the vocabulary set (if it has symbols outside of the vocabulary set, the identity map is assumed)
		next_w=''
		for i in range(len(w)):
			if w[i] in self.rule:
				next_w+=self.rule[w[i]]
			else:
				next_w+=w[i]
		return next_w

	def evolution(self,w,n_iter):
		#w is a word (also called axiom in this context) over the vocabulary set (if it has symbols outside of the vocabulary set, the identity map is assumed)
		#n_iter, the amount of iterations
		print('calculate DNA...')
		for n in range(n_iter):
			w=self.iter_step(w)
		return w

class Tree_interpreter():
	#this interpreter turns a string with the standard vocabulary {'X','F','-','+','[',']'} into a set of line segments by the 
	#rule of turtle walk
	def __init__(self,plus_angle=np.pi/8,minus_angle=-np.pi/8,randomize_angle=False):
		self.p_angle=plus_angle
		self.m_angle=minus_angle
		self.randomize=randomize_angle
		self.ex=np.array([1,0])

	def angle_mutation(self):
		self.p_angle*=np.random.rand()/2+0.75
		self.m_angle*=np.random.rand()/2+0.75

	def render(self,w):
		#w must be a word over the standard alphabet
		#returns a list of line segment objects represented by the start coordinates (x_s,y_s) and end coordinates (x_e,y_e)
		#the starting point is (0,0) and starting direction is (0,1)
		print('calculate line segments ...')
		starting_root=np.array([0,0])
		starting_angle=np.pi/2
		line_segments,_=self.rendering_recursion(starting_root,starting_angle,w)
		return line_segments

	def rendering_recursion(self,root,root_angle,w):
		current_angle=root_angle
		current_root=root
		line_segments=[]
		i=0
		length_w=len(w)
		while i<=length_w-1:
			if w[i]=='[':
				additional_line_segments, n=self.rendering_recursion(current_root,current_angle,w[i+1:])
				line_segments+=additional_line_segments
				i+=n
			elif w[i]==']':
				return line_segments,i+1
			elif w[i]=='F':
				next_root=current_root+rotation(current_angle,self.ex)
				line_segments.append([current_root,next_root])
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


















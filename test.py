import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
import pickle
import argparse
from copy import deepcopy

from l_system import *
from utils import *



# l_system=L_system()
# l_system.update_vocabulary({'X','F','-','+','[',']'})
# # l_system.add_rule('X', 'F+[[X]-X]-F[-FX]+X',verbose=True)
# # l_system.add_rule('X', '[-FX]',verbose=True)
# l_system.add_rule('X', '[-F[X]]',verbose=True)
# # l_system.add_rule('X', '-F[[-F]]',verbose=True)

# depth=3
# # l_system.add_rule('X'+str(depth), 'F[X]',verbose=True)

# TI=Depth_specific_tree_interpreter(depth=depth,plus_angles=[np.pi/8,np.pi/5,np.pi/5],minus_angles=[-np.pi/4,-np.pi/6,-np.pi/5],lengths=[1,1,1],cross_sections=[0.01,0.02,0.01],leaf_density=0.01)
# new_depth=2
# TI.modify_depth(new_depth)
# ls_list,ti_list=mutation([l_system],[TI],n_mut=2,leaf_specific=False,max_character=15)

# w,dl=l_system.evolution('X',new_depth)
# print(w)
# b=TI.render(w,dl)
# draw_branches(b,im_size=2000).show()

# raise ValueError('asdf')

n_rep=5#number of repeated independent experiments
n_gen=20#number of simulated generations 
n_trees=60#number of trees in each generation
n_sel=3#number of selected survivors from which the next generation is derived

init_depth=3#the initial depth of the L-system (could refer to the "age" of the tree)
max_depth=4#the maximal achievable depth of the L-system
leaf_specific=False#the possibility to have a different rule for the last branch at which the leaf sits
leaf_density=0#if the leaf_density is a mass density for the leaf (which can vary in its size and hence its mass)
#loop over the number of independent experiment repetitions
for u in range(n_rep):
	#gif_images=[]
	text_list=[]
	branches_for_gif=[]
	depth_specific=leaf_specific
	l_system=L_system()#initializes an L-system
	l_system.update_vocabulary({'X','F','-','+','[',']'})#introduces the vocabulary of the L-system
	init_w=l_system.sample_word(extension={'[X]','[F]'},immune={'[',']'},length=4)#samples a word from the vocabulary plus extension minus immune
	#init_w='F+[[X]-X]-F[-FX]+X'
	l_system.add_rule('X', init_w,verbose=True)#we add the random rule X->init_w
	if leaf_specific:
		init_w_depth=l_system.sample_word(extension={'[X]','[F]'},immune={'[',']'},length=4)
		l_system.add_rule('X'+str(init_depth), init_w_depth,verbose=True)#we add the additional random rule X#->init_w_depth
	#the depth_specific renderer makes the paramters angles and length of the branches depth_dependent
	TI=Depth_specific_tree_interpreter(depth=init_depth,plus_angles=[np.pi/8]*init_depth,minus_angles=[-np.pi/8]*init_depth,lengths=[1]*init_depth,cross_sections=[0.1]*init_depth,leaf_radius=0.2,leaf_density=leaf_density)
	#generates the mutants of the next generation by mutating the l-system and the tree interpreter of all survivors
	ls_list,ti_list=mutation([l_system],[TI],n_mut=n_trees,leaf_specific=leaf_specific,max_character=15,max_depth=max_depth)
	best_score=-1e10
	#----the loop over all generations
	for g in range(n_gen):
		trees=[]#here we collect the branch-topologyof each tree
		#loop overa ll the trees in the population
		w_list=[]
		dl_list=[]
		for i in range(n_trees):
			#evolution creates the rerucsive word starting from the axiom 'X' and iterating over the tree's depth
			#we also store the iteration depth of each letter at which it was generated
			w,dl=ls_list[i].evolution('X',ti_list[i].depth,depth_specific=depth_specific)
			w_list.append(w)
			dl_list.append(dl)
			branches=ti_list[i].render(w,dl)#renders the word to the list of tree branches
			trees.append(branches)
		#---SELECTION---
		#evaluates each tree using a score function and selects the best n_sel surivors
		selected_ls, selected_ti,max_score,best_branches=selection(ls_list,ti_list,w_list,dl_list,trees,n_sel=1)
		#selected_ls, selected_ti,max_score,best_branches=selection(ls_list,ti_list,n_sel=n_sel,depth_specific=depth_specific)
		#----MUTATION---
		ls_list,ti_list=mutation(selected_ls,selected_ti,n_mut=n_trees,leaf_specific=leaf_specific,max_character=15,max_depth=max_depth)
		best_ls=selected_ls[0]
		best_ti=selected_ti[0]
		
		#-----print interesting properties of the generation----
		print('Epoche='+str(g)+'---Best tree: score='+str(max_score)+', depth='+str(best_ti.depth)+', gene_code='+str(best_ls.rule['X']))
		w,dl=best_ls.evolution('X',best_ti.depth,depth_specific=depth_specific)
		#l_system=best_ls
		#TI=best_ti
		if max_score>best_score:
			best_score=max_score
			get_morph_score(best_branches,best_ti,w,dl,verbose=True)
			#im=draw_trees_from_branches(trees,im_size=3000)
			#im.save('images/branch_tree_gen='+str(g)+'_score='+str(best_score)[0:4]+'.png'))
			_,_,average_leaf_height=get_leaf_projection(best_branches,0)
			text=['generation: '+str(g),'score: '+str(best_score)[0:5], 'L-gene: '+str(best_ls.rule['X']), 'leaf radius: '+str(best_ti.leaf_radius)[0:5], 'height: '+str(average_leaf_height)[0:5],'depth: '+str(best_ti.depth)]
			text_list.append(text)
			#im_best=draw_branches(best_branches,im_size=500,text=text)
			#im_best.save('images/best_tree_gen='+str(g)+'_score='+str(best_score)[0:5]+'.png')
			#gif_images.append(im_best)
			branches_for_gif.append(best_branches)
		if g==n_gen-1:
			text=['generation: '+str(g),'score: '+str(best_score)[0:5], 'L-gene: '+str(best_ls.rule['X']), 'leaf radius: '+str(best_ti.leaf_radius)[0:5],'height: '+str(average_leaf_height)[0:5],'depth: '+str(best_ti.depth)]
			#im_best=draw_branches(best_branches,im_size=500,text=text)
			for i in range(10):
				#gif_images.append(im_best)
				branches_for_gif.append(best_branches)
				text_list.append(text)
	#draw_branches(TI.render(w,dl,starting_root=np.array([0,0]),starting_angle=np.pi/2),im_size=1000,text=text).show()
	gif_images=draw_sequence_of_branches(branches_for_gif,text_list,im_size=500,draw_leafs=False,draw_stress=True, max_stress_only=False)
	make_gif(gif_images,save_path='gifs/tree_evolution_text_score='+str(best_score)[0:5]+'.gif',duration=300)






# for u in range(1):
# 	gif_images=[]
# 	depth=2
# 	n_gen=30
# 	TI=Depth_specific_tree_interpreter(depth=depth,plus_angles=[np.pi/8,np.pi/5],minus_angles=[-np.pi/8,-np.pi/6],lengths=[1,1],cross_sections=[0.1,0.1])
# 	n_trees=200
# 	for g in range(n_gen):
# 		ls_list=[]
# 		ti_list=[]
# 		trees=[]
# 		scores=[]
# 		w,dl=l_system.evolution('X',TI.depth)
# 		branches=TI.render(w,dl)
# 		best_score=get_score(branches,TI,w,dl)
# 		trees.append(branches)
# 		scores.append(best_score)
# 		ls_list.append(l_system)
# 		ti_list.append(TI)
# 		if g==0:
# 			text=['generation: '+str(g),'score: '+str(best_score)[0:5], 'L-gene: '+str(l_system.rule['X'])]
# 			im=draw_branches(branches,im_size=500,text=text)
# 			gif_images.append(im)
# 		for i in range(n_trees-1):
# 			ls=deepcopy(l_system)
# 			ti=deepcopy(TI)

# 			ti.angle_and_size_mutation()
# 			ti.depth_mutation(p=0.2,max_depth=4,min_depth=2)
# 			ls.general_mutation('X',p=0.05,immune={'[',']'},words={'[X]','[F]'})

# 			w,dl=ls.evolution('X',ti.depth)
# 			branches=ti.render(w,dl)

# 			trees.append(branches)
# 			scores.append(get_score(branches,ti,w,dl))
# 			ls_list.append(ls)
# 			ti_list.append(ti)

# 		selected_ls, selected_ti,best_score=selection(ls_list,ti_list,n_sel=1)
# 		# max_idx=np.argmax(scores)
# 		# max_ti=ti_list[max_idx]
# 		# max_ls=ls_list[max_idx]
# 		# max_tree=trees[max_idx]

# 		print('Epoche='+str(g)+'---Best tree: score='+str(scores[max_idx])+', depth='+str(max_ti.depth)+', gene_code='+str(max_ls.rule['X']))
# 		w,dl=max_ls.evolution('X',max_ti.depth)
# 		get_score(max_tree,max_ti,w,dl,verbose=True)

# 		# l_system=max_ls
# 		# TI=max_ti
		
# 		l_system=selected_ls[0]
# 		TI=selected_ti[0]
# 		if scores[max_idx]>best_score:
# 			best_score=scores[max_idx]
# 			#im=draw_trees_from_branches(trees,im_size=3000)
# 			#im.save('images/branch_tree_gen='+str(g)+'_score='+str(best_score)[0:4]+'.png')
# 			im_best=draw_branches(max_tree,im_size=500,text=['generation: '+str(g),'score: '+str(best_score)[0:5], 'L-gene: '+str(max_ls.rule['X'])])
# 			#im_best.save('images/best_tree_gen='+str(g)+'_score='+str(best_score)[0:5]+'.png')
# 			gif_images.append(im_best)
# 		if g==n_gen-1:
# 			text=['generation: '+str(g),'score: '+str(best_score)[0:5], 'L-gene: '+str(max_ls.rule['X'])]
# 			im_best=draw_branches(max_tree,im_size=500,text=text)
# 			for i in range(10):
# 				gif_images.append(im_best)
# 	#draw_branches(TI.render(w,dl,starting_root=np.array([0,0]),starting_angle=np.pi/2),im_size=1000,text=text).show()
# 	make_gif(gif_images,save_path='gifs/tree_evolution_text_score='+str(best_score)[0:5]+'.gif',duration=300)





		








# l_system.add_rule('X0', 'F+[[X]-X]-F[-FX]+X',verbose=True)
# l_system.add_rule('X1', 'F+[[X]-X]-F[-FX]+X',verbose=True)
# l_system.add_rule('X2', 'F+[[X]-X]-F[-FX]+X',verbose=True)
# l_system.add_rule('X3', 'F-[[X]+X]',verbose=True)
# l_system.add_rule('X4', 'F-[[X]+X]F',verbose=True)
# l_system.add_rule('X5', 'F-[[X]+X]F',verbose=True)
# w,_=l_system.evolution('X',6,depth_specific=False)
# segments=tree_interpreter.render(w)
# im=draw_tree(segments,im_size=800)
# im.show()


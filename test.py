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

l_system=L_system()
l_system.update_vocabulary({'X','F','-','+','[',']'})
# l_system.add_rule('X', 'F+[[X]-X]-F[-FX]+X',verbose=True)
l_system.add_rule('X', 'F[-X]+F',verbose=True)

for u in range(1):
	gif_images=[]
	depth=2
	n_gen=20
	TI=Depth_specific_tree_interpreter(depth=depth,plus_angles=[np.pi/8,np.pi/5],minus_angles=[-np.pi/8,-np.pi/6],lengths=[1,1],cross_sections=[0.1,0.1])
	n_trees=200
	for g in range(n_gen):
		ls_list=[]
		ti_list=[]
		trees=[]
		scores=[]
		w,dl=l_system.evolution('X',TI.depth)
		branches=TI.render(w,dl)
		best_score=get_score(branches)
		trees.append(branches)
		scores.append(get_score(branches))
		ls_list.append(l_system)
		ti_list.append(TI)
		if g==0:
			gif_images.append(draw_branches(branches,im_size=500,text=['generation: '+str(g),'score: '+str(best_score)[0:5]]))
		for i in range(n_trees-1):
			ls=deepcopy(l_system)
			ti=deepcopy(TI)

			ti.angle_and_size_mutation()
			ti.depth_mutation(p=0.2,max_depth=5,min_depth=2)
			ls.general_mutation('X',p=0.05,immune={'[',']'},words={'[X]','[F]'})

			w,dl=ls.evolution('X',ti.depth)
			branches=ti.render(w,dl)

			trees.append(branches)
			scores.append(get_score(branches))
			ls_list.append(ls)
			ti_list.append(ti)

		max_idx=np.argmax(scores)
		print('Epoche='+str(g)+'---Best tree: score='+str(scores[max_idx])+', depth='+str(ti_list[max_idx].depth)+', gene_code='+str(ls_list[max_idx].rule['X']))
		#get_score(trees[max_idx],verbose=True)
		l_system=ls_list[max_idx]
		TI=ti_list[max_idx]
		if scores[max_idx]>best_score:
			best_score=scores[max_idx]
			# im=draw_trees_from_branches(trees,im_size=3000)
			# im.save('images/branch_tree_gen='+str(g)+'_score='+str(best_score)[0:4]+'.png')
			im_best=draw_branches(trees[max_idx],im_size=500,text=['generation: '+str(g),'score: '+str(best_score)[0:5]])
			#im_best.save('images/best_tree_gen='+str(g)+'_score='+str(best_score)[0:5]+'.png')
			gif_images.append(im_best)
		if g==n_gen-1:
			im_best=draw_branches(trees[max_idx],im_size=500,text=['generation: '+str(g),'score: '+str(best_score)[0:5]])
			for i in range(10):
				gif_images.append(im_best)
	make_gif(gif_images,save_path='gifs/tree_evolution_text_score='+str(best_score)[0:5]+'.gif',duration=300)

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


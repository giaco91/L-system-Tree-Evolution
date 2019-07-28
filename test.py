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
l_system.add_rule('X', 'F+[[X]-X]-F[-FX]+X',verbose=True)
# l_system.add_rule('X', '[FX]',verbose=True)
l_system.add_rule('F', 'FF',verbose=True)
# print(l_system.vocabulary)
# print(l_system.rule)
# l_system.add_mutation('X',1,words={'[]'},immune={'[',']'})
# print(l_system.rule)
tree_interpreter=Tree_interpreter(plus_angle=np.pi/8,minus_angle=-np.pi/6,randomize_angle=False)

n_trees=64
trees=[]
w=l_system.evolution('X',6)
trees.append(tree_interpreter.render(w))
for i in range(n_trees-1):
	ls=deepcopy(l_system)
	ti=deepcopy(tree_interpreter)
	ls.point_mutation('X',0.01,immune={'[',']'})
	ls.perm_mutation('X',0.02)
	ls.loss_mutation('X',0.01,immune={'[',']'})
	ls.add_mutation('X',0.01,immune={'[',']'})
	w=ls.evolution('X',6)
	ti.angle_mutation()
	trees.append(ti.render(w))
	print(ls.rule)
im=draw_trees(trees,im_size=3000)
im.show()
im.save('images/offspring.png')

# segments=tree_interpreter.render(w)
# im=draw_tree(segments,im_size=800)
# im.show()


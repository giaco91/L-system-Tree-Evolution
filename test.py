import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import random
import pickle
import argparse

from l_system import *
from utils import *

l_system=L_system()
l_system.update_vocabulary({'X','F','-','+','[',']'})
# l_system.add_rule('X', 'F+[[X]-X]-F[-FX]+X',verbose=True)
l_system.add_rule('X', 'F+[X-X-F[-FX]+X]',verbose=True)
l_system.add_rule('F', 'FF',verbose=True)
# print(l_system.vocabulary)
# print(l_system.rule)
l_system.point_mutation('X',0.1,immune={'[',']'})
l_system.perm_mutation('X',0.1)
l_system.loss_mutation('X',0.1,,immune={'[',']'})


w=l_system.evolution('X',2)

tree_interpreter=Tree_interpreter(plus_angle=np.pi/8,minus_angle=-np.pi/6,randomize_angle=False)
segments=tree_interpreter.render(w)
im=draw_tree(segments,im_size=800)
im.show()


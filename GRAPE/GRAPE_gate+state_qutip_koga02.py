from qutip import *
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
import os
import sys
#このファイルとQ_moduleを別々のフォルダにに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append("C:/Users/yuta/.ipython/profile_default/機械学習/サブpy")
from Q_module_new02 import*
from Q_module_grape_qutip_koga02 import*
import time
start = time.time()

gf = Grape_funcs_for_single_wire()
gf.main()

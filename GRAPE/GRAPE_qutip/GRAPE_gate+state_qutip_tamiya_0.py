from qutip import *
from scipy import *
from numpy import *
import matplotlib.pyplot as plt
import os
import sys
#このファイルとQ_moduleを別々のフォルダに入れる場合は、ここにmoduleのフォルダ名を入れる
sys.path.append(r"C:\data&prog\koga\Github\king_branch\GRAPE\サブpy")
from Q_module_new02 import*
from Q_module_grape_qutip_tamiya02 import*
import time
start = time.time()

gf=Grape_funcs_for_cross_wire()
gf.main()
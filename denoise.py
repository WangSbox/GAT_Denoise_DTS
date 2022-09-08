# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import shutil
# os.system("python ./gcndenoise/gcndenoise.py --epoch 150 --bs 2048 --nw 8 --lr 1e-3 --lc 20 --trn 3 --tesn 10 --tgac 0.85 --ts 1e-3 --ab 1 --round 1 --dist 1 --th 0.005")
shutil.copy('./gcndenoise/gcndenoisemodeltrain.py','./model')
shutil.copy('./denoise.py','./model')
os.system("python ./gcndenoise/gcndenoise.py --epoch 50 --bs 256 --nw 4 --lr 1e-3 --lc 4 --trn 4 --tesn 5 --tgac 0.68 --ts 1e-3 --ab 0 --round 1 --dist 0 --th 0.003")

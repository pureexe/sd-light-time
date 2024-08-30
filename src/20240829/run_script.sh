#version 0 - lr 5e-4 - LOSS NAN
bin/py src/20240829/train.py -lr 5e-4

#version 1 -lr 1e-4
bin/py src/20240829/train.py-lr 1e-4

#version 2 -lr 5e-5
bin/py src/20240829/train.py -lr 5e-5

#version 3 -lr 1e-5
bin/py src/20240829/train.py -lr 1e-5

#version 4 -lr 1e-4 / mult10
bin/py src/20240829/train.py -lr 1e-4 -gm 10

#version 5 -lr 1e-4 / mult10
bin/py src/20240829/train.py -lr 1e-4 -gm 100
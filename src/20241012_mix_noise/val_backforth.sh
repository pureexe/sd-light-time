CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_backforth/val_ddim_mix.py -s '1.0' -g '1.25,1.5,1.75,2.25,2.5,2.75,4.0,6.0'
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.75' -g '1.25,1.5,1.75,2.25,2.5,2.75,4.0,6.0'
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.25' -g '1.25,1.5,1.75,2.25,2.5,2.75,4.0,6.0'
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.1,0.2' -g '1.25'


# v2
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0' -g '1.0'


CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '1.0' -g '1.0'


CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0,0.1,0.2' -g '1.0'
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.3,0.4' -g '1.0'
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.5,0.6' -g '1.0'
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.7,0.8,0.9' -g '1.0'


CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0,0.1,0.2' -g '1.25,1.5,1.75,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0'
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.3,0.4' -g '1.25,1.5,1.75,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0'
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.5,0.6,0.7' -g '1.25,1.5,1.75,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0'
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.8,0.9,1.0' -g '1.25,1.5,1.75,2.25,2.5,2.75,3.0,4.0,5.0,6.0,7.0'

#v7
CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0,0.1,0.2' -g '1.25,1.5,1.75'
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.3,0.4' -g '1.25,1.5,1.75'
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.5,0.6,0.7' -g '1.25,1.5,1.75'
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.8,0.9,1.0' -g '1.25,1.5,1.75'

CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0,0.1,0.2' -g '2,2.25,2.5,2.75'
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.3,0.4' -g '2,2.25,2.5,2.75'
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.5,0.6,0.7' -g '2,2.25,2.5,2.75'
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.8,0.9,1.0' -g '2,2.25,2.5,2.75'

#v17
#CUDA_VISIBLE_DEVICES=0 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0,0.1,0.2' -g '3.0,4.0,5.0,6.0,7.0'
CUDA_VISIBLE_DEVICES=1 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.0,0.3,0.4' -g '3.0,4.0,5.0,6.0,7.0'
CUDA_VISIBLE_DEVICES=2 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.1,0.5,0.6,0.7' -g '3.0,4.0,5.0,6.0,7.0'
CUDA_VISIBLE_DEVICES=3 bin/py src/20241012_backforth/val_ddim_mix.py -s '0.2,0.8,0.9,1.0' -g '3.0,4.0,5.0,6.0,7.0'




gpu
bin/v100shell
cd src/20250217_albedo_optimization_v3/

# 99483
python albedo_optimization.py -lra 1e-2 -lrs 1e-2

# 99485
python albedo_optimization.py -lra 1e-2 -lrs 1e-3

# 99486
python albedo_optimization.py -lra 1e-2 -lrs 1e-4

# 99487
python albedo_optimization.py -lra 1e-2 -lrs 1e-5

# 99488
python albedo_optimization.py -lra 1e-3 -lrs 1e-2

# 99489
python albedo_optimization.py -lra 1e-3 -lrs 1e-3

# 99490
python albedo_optimization.py -lra 1e-3 -lrs 1e-4

# 99491
python albedo_optimization.py -lra 1e-3 -lrs 1e-5

###########################
# 99492 
python albedo_optimization.py -lra 1e-4 -lrs 1e-2

# 99494
python albedo_optimization.py -lra 1e-4 -lrs 1e-3

# 99495
python albedo_optimization.py -lra 1e-4 -lrs 1e-4

# 99496
python albedo_optimization.py -lra 1e-4 -lrs 1e-5

# 99497
python albedo_optimization.py -lra 1e-5 -lrs 1e-2

# 99498
python albedo_optimization.py -lra 1e-5 -lrs 1e-3

# 99500
python albedo_optimization.py -lra 1e-5 -lrs 1e-4

# 99501
python albedo_optimization.py -lra 1e-5 -lrs 1e-5

############## RUN ALBEDO
python run_albedo_optimization.py -i 0 -t 1

python run_albedo_optimization.py -t 32 -i 0
python run_albedo_optimization.py -t 32 -i 1
python run_albedo_optimization.py -t 32 -i 2
python run_albedo_optimization.py -t 32 -i 3
python run_albedo_optimization.py -t 32 -i 4
python run_albedo_optimization.py -t 32 -i 5
python run_albedo_optimization.py -t 32 -i 6
python run_albedo_optimization.py -t 32 -i 7
python run_albedo_optimization.py -t 32 -i 8
python run_albedo_optimization.py -t 32 -i 9
python run_albedo_optimization.py -t 32 -i 10
python run_albedo_optimization.py -t 32 -i 11
python run_albedo_optimization.py -t 32 -i 12
python run_albedo_optimization.py -t 32 -i 13
python run_albedo_optimization.py -t 32 -i 14
python run_albedo_optimization.py -t 32 -i 15
python run_albedo_optimization.py -t 32 -i 16
python run_albedo_optimization.py -t 32 -i 17
python run_albedo_optimization.py -t 32 -i 18
python run_albedo_optimization.py -t 32 -i 19
python run_albedo_optimization.py -t 32 -i 20
python run_albedo_optimization.py -t 32 -i 21
python run_albedo_optimization.py -t 32 -i 22
python run_albedo_optimization.py -t 32 -i 23
python run_albedo_optimization.py -t 32 -i 24
python run_albedo_optimization.py -t 32 -i 25
python run_albedo_optimization.py -t 32 -i 26
python run_albedo_optimization.py -t 32 -i 27
python run_albedo_optimization.py -t 32 -i 28
python run_albedo_optimization.py -t 32 -i 29
python run_albedo_optimization.py -t 32 -i 30
python run_albedo_optimization.py -t 32 -i 31
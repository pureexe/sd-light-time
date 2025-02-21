gpu
bin/v100shell
cd src/20250217_albedo_optimization_v4/

# 99655 
python albedo_optimization.py -lra 1e-2 -lrs 1e-2

# 99656
python albedo_optimization.py -lra 1e-2 -lrs 1e-3

# 99657
python albedo_optimization.py -lra 1e-2 -lrs 1e-4

# 99658
python albedo_optimization.py -lra 1e-2 -lrs 1e-5

# 99659
python albedo_optimization.py -lra 1e-3 -lrs 1e-2

# 99660
python albedo_optimization.py -lra 1e-3 -lrs 1e-3

# 99661
python albedo_optimization.py -lra 1e-3 -lrs 1e-4

# 99662
python albedo_optimization.py -lra 1e-3 -lrs 1e-5

###########################
# 99663 
python albedo_optimization.py -lra 1e-4 -lrs 1e-2

# 99664
python albedo_optimization.py -lra 1e-4 -lrs 1e-3

# 99665
python albedo_optimization.py -lra 1e-4 -lrs 1e-4

# 99666
python albedo_optimization.py -lra 1e-4 -lrs 1e-5

# 99667
python albedo_optimization.py -lra 1e-5 -lrs 1e-2

# 99668
python albedo_optimization.py -lra 1e-5 -lrs 1e-3

# 99669
python albedo_optimization.py -lra 1e-5 -lrs 1e-4

# 99670
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
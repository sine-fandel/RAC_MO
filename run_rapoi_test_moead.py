import os

run = [0, 10, 20, 30, 40, 50, 60, 70, 79]

for i in run:
    os.system(f"sbatch rapoi_test_moead.sl {i}")

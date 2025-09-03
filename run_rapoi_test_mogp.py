import os

run = [0, 10, 20, 30, 41, 50]

for i in run:
    os.system(f"sbatch rapoi_test_mogp.sl {i}")


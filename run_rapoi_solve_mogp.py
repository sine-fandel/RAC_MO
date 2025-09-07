import os

run = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in run:
    os.system(f"sbatch rapoi_mogp.sl {i}")

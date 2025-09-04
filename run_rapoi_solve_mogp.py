import os

run = [0]

for i in run:
    os.system(f"sbatch rapoi_mogp.sl {i}")

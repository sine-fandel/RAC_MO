import os

run = [0, 9, 19, 29, 39, 49, 59, 69]

for i in run:
    os.system(f"sbatch rapoi_eval_mogp.sl {i}")
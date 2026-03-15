import os

run = range(30)
# gens = [0, 9, 29, 39, 49, 59, 69]
gens = [69]

for i in run:
    for j in gens:
        os.system(f"sbatch rapoi_eval_nsgp2.sl {i} {j}")

import os

run = range(30)
gens = [69]

for i in run:
    for j in gens:
        os.system(f"sbatch rapoi_eval_nsgp2_llm.sl {i} {j}")

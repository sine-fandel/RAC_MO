import os

# run = [0, 9, 19, 29, 39, 49, 59, 69]
run = range(30)

gens = [69]

for i in run:
    for j in gens:
        os.system(f"sbatch rapoi_eval_moead.sl {i} {j}")


# for i in run:
# for j in gens:
#     os.system(f"sbatch rapoi_eval_moead.sl {0} {j}")

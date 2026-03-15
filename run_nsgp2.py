import os

run = range(30)

for i in run:
    os.system(f"sbatch rapoi_nsgp2.sl {i}")



import os

run = range(30)

for i in run:
    os.system(f"sbatch rapoi_nsgp2_llm.sl {i}")



# os.system(f"sbatch rapoi_nsgp2_llm.sl 0")


import os

idx = 0
for pdb_dir in os.listdir("../data/steered/pdbs"):
  pdb_path = os.path.join("../data/steered/pdbs", pdb_dir)
  os.system(f"python run_parakeet.py --pdb-dir {pdb_path} --mrc-dir . --exposure 4000 -n 1 -m 250")
  os.system(f"mv 000000.mrc ../data/steered/mrc/{pdb_dir+'.mrc'}")
  os.system(f"mv 000000.yaml ../data/steered/mrc/{pdb_dir+'.yaml'}")



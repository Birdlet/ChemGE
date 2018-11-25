from optimizer import *

with open("./sample/head_100_zinc.smi") as f:
    smiles = f.read().split("\n")
smiles.pop()

opt = optimizerJ()
opt.optimize(smiles, generation=30, log="log.csv")
    

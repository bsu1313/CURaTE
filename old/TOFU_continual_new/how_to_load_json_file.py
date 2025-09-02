from datasets import load_dataset
import json

def load_split(name, cache):
    return load_dataset("json", data_files=f"/home/work/data/seyun_workspace_home/cache_LTE/TOFU/{name}.json", split="train", cache_dir=cache)
    
    
        
name = "forget10_perturbed"
forget_per = load_split(name, "/home/work/data/seyun_workspace/cache_LTE/")

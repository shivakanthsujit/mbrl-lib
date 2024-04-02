from collections import defaultdict
import glob
from pprint import pprint
import numpy as np
from pathlib import Path

import pandas as pd

files = glob.glob("logs/planet/default/**/**/eval.npz")
files = sorted(files)

data = defaultdict(list)

for fname in files:
    parts = Path(fname).parts
    env_name = parts[-3]
    splits = env_name.split("-")
    sparse = splits[-1]
    env = "-".join(splits[:-1])
    x = np.load(fname)["rews"]
    data[env_name].append(x)

env_map = {
    "polygon-obs": "Hallways",
    "room-multi-passage": "Rooms",
    "room-spiral": "Spiral"
}

sparse_map = {
    "True": "Sparse",
    "False": "Dense"
}
print_data = {}
for env_name, env_data in data.items():
    splits = env_name.split("-")
    sparse = splits[-1]
    env = "-".join(splits[:-1])
    name = f"{env_map[env]} ({sparse_map[sparse]})"
    env_data = np.array(env_data)
    print_data[name] = f"{env_data.mean():.2f} +- {env_data.std():.2f}"

df = pd.DataFrame(print_data.items(), columns=["Env", "Score"])

# mean_data = defaultdict(lambda: defaultdict())
# for sparse, sparse_dict in data.items():
#     for env, env_list in sparse_dict.items():
#         mean_data[sparse][env] = np.mean(env_list)

# std_data = defaultdict(lambda: defaultdict())
# for sparse, sparse_dict in data.items():
#     for env, env_list in sparse_dict.items():
#         std_data[sparse][env] = np.std(env_list)

pprint(mean_data)
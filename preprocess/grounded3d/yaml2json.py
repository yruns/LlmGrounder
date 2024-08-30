import yaml
import json
import glob
import os

ROOT = "../../data/processed/scannet200/"

for filename in glob.glob(os.path.join(ROOT, '*.yaml')):
    data = yaml.load(open(filename), Loader=yaml.FullLoader)
    json.dump(data, open(filename.replace(".yaml", ".json"), "w"), indent=4)

    print("Converted {} to {}".format(filename, filename.replace(".yaml", ".json")))


import json
import yaml
import subprocess

#  fast individual: pid 1 for participant "58c54d6d2775404a9c3a3cde65c32a71"
#  slow individual: pid 38 for participant "ece1f226b161426aafd433aa0e933b5d"

experiment = {
    "name": "fit_fast",
    "pid": 1,
    "optimize": ["T", "m", "tau", "delta"],
    "shared": [],
    "default": {"T": 0.3, "m": 0.2, "tau": 0, "delta": 0}
}

params = {}
params["pid"] = {"_type": "choice", "_value": [experiment['pid']]}
params["T"] = {"_type":"quniform","_value":[0.1, 0.5, 0.01]} if "T" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["T"]]}
params["m"] = {"_type":"quniform","_value":[0.1, 0.3, 0.01]} if "m" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["m"]]}
params["tau"] = {"_type":"quniform","_value":[0.01, 0.1, 0.01]} if "tau" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["tau"]]}
params["delta"] = {"_type":"quniform","_value":[0.1, 1.0, 0.1]} if "delta" in experiment['optimize'] else {"_type": "choice", "_value": [experiment['default']["delta"]]}

with open(f"params.json", "w") as outfile:
    json.dump(params, outfile)


config = {}
config['experimentName'] = experiment['name'] + str(experiment['pid'])
config['searchSpaceFile'] = "params.json"
config['trialCommand'] = "python fit_params.py"
config['trialCodeDirectory'] = "."
config['trialConcurrency'] = 8
config['maxExperimentDuration'] = "72h"
config['maxTrialNumber'] = 1000
config['tuner'] = {"name": "TPE", "classArgs": {"optimize_mode": "minimize"}}
config['trainingService'] = {"platform": "local", "useActiveGpu": False}

with open(f'config.yaml', 'w') as f:
    yaml.safe_dump(config, f, default_flow_style=False)

cmd_str = f"nnictl create --config config.yaml --port {8090+experiment['pid']}"
subprocess.run(cmd_str, shell=True)
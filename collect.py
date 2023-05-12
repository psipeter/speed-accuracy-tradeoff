import json
import yaml
import subprocess
import re

cmd_str = "nnictl experiment list"
experiment_string = str(subprocess.run(cmd_str, shell=True, capture_output=True, text=True))

id_str = "Id: "
res = [i.start() for i in re.finditer(id_str, experiment_string)]

experiments = []
for i in range(len(res)):
    experiments.append(experiment_string[res[i]+4: res[i]+12])

for exp in experiments:
    cmd_str = f"nnictl experiment export {exp} --type json --filename data/{exp}.json"
    subprocess.run(cmd_str, shell=True)
    f = open(f'data/{exp}.json')
    result = json.load(f)
    min_loss = 1000
    pid = result[0]['parameter']['pid']
    best_args = {}
    for trial in result:
        loss = int(trial['value'])
        if loss <= min_loss:
            min_loss = loss
            best_args = trial['parameter']
            best_args['loss'] = loss

    with open(f"data/{pid}.json", "w") as outfile:
        json.dump(best_args, outfile)
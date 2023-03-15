import json

dataset1 = json.loads(open("dataset4.json").read())
dataset2 = json.loads(open("test.json").read())

for i in range(len(dataset2["X"])):
    dataset1["X"].append(dataset2["X"][i])
    dataset1["Y"].append(dataset2["Y"][i])

with open("dataset5.json", "w") as fp:
    json.dump(dataset1,fp) 

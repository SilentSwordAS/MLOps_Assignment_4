import pandas as pd
from sklearn.model_selection import train_test_split
import random
import sys

data = pd.read_csv("augmented_train_v1.csv")

train, test = train_test_split(data, test_size=0.2, random_state=42,  stratify=data["species"])

clean_data, poisoned_data = train_test_split(train, test_size=float(sys.argv[1]), random_state=42, stratify=train["species"])

sepal_length = []
petal_length = []
sepal_width = []
petal_width = []

for i in range(len(poisoned_data)):
  sepal_length.append(random.uniform(min(poisoned_data["sepal_length"])+10,max(poisoned_data["sepal_length"])+10))
  sepal_width.append(random.uniform(min(poisoned_data["sepal_width"])+10,max(poisoned_data["sepal_width"])+10))
  petal_length.append(random.uniform(min(poisoned_data["petal_length"])+10,max(poisoned_data["petal_length"])+10))
  petal_width.append(random.uniform(min(poisoned_data["petal_width"])+10,max(poisoned_data["petal_width"])+10))

poisoned_data["sepal_length"] = sepal_length
poisoned_data["sepal_width"] = sepal_width
poisoned_data["petal_length"] = petal_length
poisoned_data["petal_width"] = petal_width

train = pd.concat([clean_data, poisoned_data])

train.to_csv(f"poisoned_data_{sys.argv[1]}.csv", index=False)
test.to_csv("test.csv", index=False)



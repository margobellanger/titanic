import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

print(train_data.head())
labels = train_data["Survived"]

# todo: validation set

data = [train_data, test_data]

for dataset in data:
    mean = train_data["Age"].mean()
    std = test_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)
train_data["Age"].isnull().sum()

# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
features = ["Sex", "Age", "Pclass"]

train = pd.get_dummies(train_data[features])
test = pd.get_dummies(test_data[features])

random_forest = RandomForestClassifier(n_estimators=100, oob_score=True)
random_forest.fit(train, labels)
Y_prediction = random_forest.predict(test)

random_forest.score(train, labels)

acc_random_forest = round(random_forest.score(train, labels) * 100, 2)
print(round(acc_random_forest, 2, ), "%")

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': Y_prediction})
output.to_csv('titanic.csv', index=False)

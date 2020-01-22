from sklearn.ensemble import RandomForestClassifier

from preprocessor import get_processed_data

X_train, Y_train, X_test = get_processed_data()

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(acc_random_forest)
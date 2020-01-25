from sklearn import linear_model
from preprocessor import get_processed_data

X_train, Y_train, X_test = get_processed_data()

sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, Y_train)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print(acc_sgd)

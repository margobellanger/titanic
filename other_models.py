from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from preprocessor import get_processed_data

X_train, Y_train, X_test = get_processed_data()


def svc(X_train, Y_train):
    linear_svc = LinearSVC()
    linear_svc.fit(X_train, Y_train)
    Y_pred = linear_svc.predict(X_test)
    acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
    print(acc_linear_svc)


def knn(X_train, Y_train):
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

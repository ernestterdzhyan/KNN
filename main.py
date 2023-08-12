from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from knn import KNN
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

my_knn = KNN(k=3)
my_knn.fit(X_train, y_train)

my_predictions = my_knn.predict(X_test)

sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)
sklearn_predictions = sklearn_knn.predict(X_test)

accuracy_my_knn = accuracy_score(y_test, my_predictions)
accuracy_sklearn_knn = accuracy_score(y_test, sklearn_predictions)

print(f"Accuracy of my KNN: {accuracy_my_knn:.4f}")
print(f"Accuracy of sklearn's KNN: {accuracy_sklearn_knn:.4f}")

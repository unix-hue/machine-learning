import numpy as np
from sklearn.svm import LinearSVC

n, m, l, k = map(int, input().split())

X = np.ndarray((n, m + l), dtype = "float")
for i in range(n):
    X[i] = np.array(input().split())

y = X[:, m + k].astype(int)
y1 = X[:, m+1].astype(int)#Check
X = np.delete(X, m, axis=1)
X = np.delete(X, m, axis=1)#Check

clf = LinearSVC().fit(X, y)
clf1 = LinearSVC().fit(X, y1)#Check

coef = clf.coef_[0]
inter = clf.intercept_[0]
norm = np.linalg.norm(coef)

X2 = np.ndarray((n, m), dtype = "float")#Check
for i in range(len(X)):
    distance = (coef / norm) * (np.sum(coef / norm * X[i]) + inter / norm)
    X2[i] = np.round(X[i] - 1.1 * distance, 1)
    print(*X2[i])

########################################################
test1 = clf.predict(X)
test2 = clf1.predict(X)
res1 = clf.predict(X2)
res2 = clf1.predict(X2)

print('Изменений в первом столбце', np.sum(np.abs(res1 - test1)))
print('Изменений во втором столбце', np.sum(np.abs(res2 - test2)))


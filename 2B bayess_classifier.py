import scipy.stats as sps
import numpy as np

n, m, k, l = np.loadtxt('input.txt', max_rows=1).astype(int)
lambda_ = np.loadtxt('input.txt', skiprows=1, max_rows=m).astype(int)
train = np.loadtxt('input.txt', skiprows=1 + m, max_rows=k)
c = train[:, -1]
train = np.delete(train, -1, axis=1)
test = np.loadtxt('input.txt', skiprows=1 + m + k)

classes, count_classes = np.unique(c, return_counts=True)
classes_aprio = count_classes / len(c)

for i in range(m):
    lambda_[i, i] = 1
penalty = np.prod(lambda_, axis=1)

means = []
for class_ in classes:
    means.append(np.mean(train[c == class_], axis=0))

covs = []
for class_ in classes:
    covs.append(np.cov(train[c == class_].T))

res = []
for i in range(l):
    norm_density = []
    
    for class_ in range(m):
        norm_density.append(sps.multivariate_normal.pdf(test[i], mean=means[class_], cov=covs[class_]))

    pred = np.argmax(classes_aprio * norm_density * penalty)
    res.append(pred)

np.savetxt('output.txt', res, fmt='%i', newline=' ')
print(*res)

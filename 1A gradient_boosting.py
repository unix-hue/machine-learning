import numpy as np

m, n, k = map(int, input().split())
X = np.ndarray((m, n + 1)).astype(int)
for i in range(m):
    X[i] = np.array(input().split()).astype(int)
y = X[:, -1]
X = np.delete(X, -1, axis=1)

h0 = np.log(np.sum(y == 1) / np.sum(y == 0))
print(np.round(h0, 3))

H = np.zeros(m) + h0
I = list(range(n))
T = []
for i in range(n):
    T.append(np.unique(X[:, i]))

for _ in range(k):
    predictions = 1 / (1 + np.exp(-H))
    r = y - predictions

    loss = float('inf')
    for l in I:
        if l == None:
            continue

        feature = X[:, l]
        sum_r = np.zeros(len(T[l]))
        count_r = np.zeros(len(T[l]))

        for t in T[l]:
            r_t = r[feature == t]
            sum_r[t] = np.sum(r_t)
            count_r[t] = len(r_t)

        mean_r = sum_r / count_r
        local_loss = np.mean(np.square(r - mean_r[feature]))

        if (local_loss < loss):
            best_regressor = l
            best_sum = sum_r
            loss = local_loss

    x = X[:, best_regressor]
    P_t = np.zeros(len(T[best_regressor]))

    for i in T[best_regressor]:
        P_t[i] = np.sum(predictions[x == i] * (1 - predictions[x == i]))

    h = best_sum / P_t
    H = H + h[x]
    I[best_regressor] = None
    print(best_regressor, *np.round(h, 3))
    
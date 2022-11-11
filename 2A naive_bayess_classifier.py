import numpy as np

n, m, k = map(int,input().split())

def predict(sample, words_in_class, words_at_all, class_count):
    class_words = words_in_class[:, sample != 0]
    all_words = words_at_all.reshape(-1, 1)
    rel_words = np.prod(class_words / all_words, axis=1).reshape(-1,1)
    return np.argmax(rel_words * class_count.reshape(-1, 1))

words_in_class = np.ones((k, m)).astype(int)
words_at_all = np.zeros(k).astype(int) + m
class_count = np.ones(k).astype(int)
result = []

sample = np.array(input().split()).astype(int)
result.append(predict(sample[:-1], words_in_class, words_at_all, class_count))

for i in range(n - 1):
    class_count[sample[-1]] += 1
    words_in_class[sample[-1], :] += sample[:-1]
    words_at_all[sample[-1]] += np.sum(sample[:-1])

    sample = np.array(input().split()).astype(int)
    result.append(predict(sample[:-1], words_in_class, words_at_all, class_count))

for value in result:
    print(value)

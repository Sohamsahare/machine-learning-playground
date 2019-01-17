from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
	if len(data) >= k:
		warnings.warn('K is set to a value less than total voting groups!')
	distances = []
	for group in data:
		for features in data[group]:
			#euclidean_distance = np.sqrt(np.sum((np.array(features) - np.array(predict))**2))
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance,group])
			
	votes = [i[1] for i in sorted(distances)[:k]]
# Counter(votes).most_common(1) => [['k',3]] => type 'k' has 3 occurences or matches
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1] / k
	return vote_result

accuracies = []

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()
for i in range(25):
	random.shuffle(full_data)

	# test data size wrt full_data
	test_size = 0.2

	# declaring data partitions
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}

	# popuulating data for training and testing
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]

	for i in train_data:
		train_set[i[-1]].append(i[:-1])
		
	for i in test_data:
		test_set[i[-1]].append(i[:-1])

	correct = 0
	total = 0

	for group in test_set:
		for data in test_set[group]:
			# output from our implementation of k_nn
			vote = k_nearest_neighbors(train_set, data, k=5)
			# comparing with the group it actually belongs to
			if group == vote:
				correct += 1
			total += 1
	print( 'Accuracy:', correct/total)
	accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))
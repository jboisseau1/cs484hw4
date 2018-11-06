import pandas as pd
import numpy as np
import random
import heapq

from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from surprise import PredictionImpossible
from surprise.prediction_algorithms.knns import SymmetricAlgo


training_set = pd.read_csv('additional_files/train.dat',delim_whitespace=True)
test_set = pd.read_csv('1540226926_0229266_test.dat',delim_whitespace=True)


class customRecommender(SymmetricAlgo):
    def __init__(self):
        SymmetricAlgo.__init__(self)
        self.k = 40
        self.min_k = 1

    def fit(self, trainset):
        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()
        return self

    def estimate(self, u, i):
        # checks if there is data for given user and movie IDs
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)
        # gets the nearest neighbors for each user
        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        total_similarity = 0
        total_ratings = 0
        current_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                total_similarity += sim
                total_ratings += sim * r
                current_k += 1

        unknown_user_rating = total_ratings / total_similarity

        return unknown_user_rating





recommender = customRecommender()
reader = Reader(rating_scale=(0, 5)) # sets min and max of rating system

# reads in training set and configes it for recommending
data = Dataset.load_from_df(training_set[['userID', 'movieID', 'rating']], reader)
trainset = data.build_full_trainset()
recommender.fit(trainset)

results = []
print('🎯 Working... 🎯')
# rates each test input
for i, test_row in test_set.iterrows():
    test_userID = test_row['userID']
    test_movieID = test_row['movieID']
    pred = recommender.predict(test_userID, test_movieID).est
    results.append(pred)


output_file = str('rating_output'+str(random.randint(1,1001))+'.dat')
file = open(output_file,'w')
for i in range(0,len(results)):
    file.write(str(results[i]) + '\n')
print('Finished! Results in ➡️',output_file)

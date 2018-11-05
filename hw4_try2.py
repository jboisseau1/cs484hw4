import pandas as pd
import numpy as np
import random
from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from surprise import PredictionImpossible
from surprise.prediction_algorithms.knns import SymmetricAlgo


training_set = pd.read_csv('additional_files/train.dat',delim_whitespace=True)
test_set = pd.read_csv('1540226926_0229266_test.dat',delim_whitespace=True)


class KNNBasic(SymmetricAlgo):
    def __init__(self, k=40, min_k=1, sim_options={}, verbose=True, **kwargs):

        SymmetricAlgo.__init__(self, sim_options=sim_options, verbose=verbose,
                               **kwargs)
        self.k = k
        self.min_k = min_k

    def fit(self, trainset):

        SymmetricAlgo.fit(self, trainset)
        self.sim = self.compute_similarities()

        return self

    def estimate(self, u, i):

        import heapq
        # checks if there is data for given user and movie IDs
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        x, y = self.switch(u, i)
        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = 0
        sum_ratings = 0
        actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        est = sum_ratings / sum_sim
        return est





algo = KNNBasic()

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0, 5))

# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(training_set[['userID', 'movieID', 'rating']], reader)

trainset = data.build_full_trainset()

algo.fit(trainset)
results = []
for i, test_row in test_set.iterrows():
    test_userID = test_row['userID']
    test_movieID = test_row['movieID']
    pred = algo.predict(test_userID, test_movieID,verbose=False).est
    results.append(pred)


print(results)
# output_file = str('test'+str(random.randint(1,1001))+'.dat')
# file = open(output_file,'w')
# for i in range(0,len(results)):
#     file.write(str(results[i]) + '\n')
# print('Finished! Results in ➡️',output_file)

import pandas as pd
import numpy as np

movie_tags = pd.read_csv('additional_files/movie_tags.dat',delim_whitespace=True)
movie_genres = pd.read_csv('additional_files/movie_genres.dat',delim_whitespace=True)
user_taggedmovies = pd.read_csv('additional_files/user_taggedmovies.dat',delim_whitespace=True)
tags = pd.read_csv('csvs/tags.csv')
directors = pd.read_csv('csvs/movie_directors.csv')
actors = pd.read_csv('csvs/movie_actors.csv')
tags.columns=['tagID','tagName']
tag_data = tags.merge(movie_tags,on='tagID')
training_set = pd.read_csv('additional_files/train.dat',delim_whitespace=True)
# print(training_set)
# ratings = pd.DataFrame(training_set.groupby('movieID')['rating'].mean())
# ratings['number_of_ratings'] = training_set.groupby('movieID')['rating'].count()

test_set = pd.read_csv('1540226926_0229266_test.dat',delim_whitespace=True)
from surprise import AlgoBase
from surprise import Dataset

import numpy as np
from six import iteritems
import heapq

from surprise import PredictionImpossible


class SymmetricAlgo(AlgoBase):
    def __init__(self, sim_options={}, verbose=True, **kwargs):

        AlgoBase.__init__(self, sim_options=sim_options, **kwargs)
        self.verbose = verbose

    def fit(self, trainset):

        AlgoBase.fit(self, trainset)

        ub = self.sim_options['user_based']
        self.n_x = self.trainset.n_users if ub else self.trainset.n_items
        self.n_y = self.trainset.n_items if ub else self.trainset.n_users
        self.xr = self.trainset.ur if ub else self.trainset.ir
        self.yr = self.trainset.ir if ub else self.trainset.ur

        return self

    def switch(self, u_stuff, i_stuff):
        if self.sim_options['user_based']:
            return u_stuff, i_stuff
        else:
            return i_stuff, u_stuff

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

        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')

        x, y = self.switch(u, i)

        neighbors = [(self.sim[x, x2], r) for (x2, r) in self.yr[y]]
        k_neighbors = heapq.nlargest(self.k, neighbors, key=lambda t: t[0])

        # compute weighted average
        sum_sim = sum_ratings = actual_k = 0
        for (sim, r) in k_neighbors:
            if sim > 0:
                sum_sim += sim
                sum_ratings += sim * r
                actual_k += 1

        if actual_k < self.min_k:
            raise PredictionImpossible('Not enough neighbors.')

        est = sum_ratings / sum_sim

        details = {'actual_k': actual_k}
        return est, details



from surprise import Reader

# from surprise import KNNBasic
import random
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

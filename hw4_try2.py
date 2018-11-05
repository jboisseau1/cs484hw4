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

from surprise import Dataset
from surprise import Reader

from surprise import KNNBasic
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
    pred = algo.predict(test_userID, test_movieID, verbose=False).est
    results.append(pred)



output_file = str('test'+str(random.randint(1,1001))+'.dat')
file = open(output_file,'w')
for i in range(0,len(results)):
    file.write(str(results[i]) + '\n')
print('Finished! Results in ➡️',output_file)

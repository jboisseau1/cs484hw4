import pandas as pd

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
ratings = pd.DataFrame(training_set.groupby('movieID')['rating'].mean())
ratings['number_of_ratings'] = training_set.groupby('movieID')['rating'].count()


import matplotlib.pyplot as plt

# ratings['rating'].hist(bins=60)

import seaborn as sns
sns.jointplot(x='rating', y='number_of_ratings', data=ratings)

movie_matrix = training_set.pivot_table(index='userID', columns='movieID', values='rating')
# print(ratings.sort_values('number_of_ratings', ascending=False).head(10))
# plt.show()

test = movie_matrix[653]
print(movie_matrix.corrwith(test))


# source: https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d

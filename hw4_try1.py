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
ratings = pd.DataFrame(training_set.groupby('movieID')['rating'].mean())
ratings['number_of_ratings'] = training_set.groupby('movieID')['rating'].count()

test_set = pd.read_csv('1540226926_0229266_test.dat',delim_whitespace=True)

movie_matrix = training_set.pivot_table(index='userID', columns='movieID', values='rating')
centered_movie_matrix = movie_matrix.subtract(movie_matrix.mean()).fillna(0)


from sklearn.metrics.pairwise import cosine_similarity
user_similarities = cosine_similarity(centered_movie_matrix.values)

K=5
# centered_movie_matrix = centered_movie_matrix.reset_index()
movie_matrix = movie_matrix.reset_index()
movieIDs = movie_matrix.columns.values[1:]

results = []
for i, test_row in test_set.iterrows():
    test_userID = test_row['userID']
    test_movieID = test_row['movieID']
    sim_index = movie_matrix.loc[movie_matrix['userID'] == test_userID].index[0]
    # print('*****',test_userID,test_movieID)
    user_sim = user_similarities[sim_index]
    N = np.sort(user_sim)[::-1]
    KNN_ratings = []
    if((test_movieID in movieIDs)):
        for NN_index in np.argsort(N):
            if(len(KNN_ratings)<K):
                if(not np.isnan(movie_matrix.loc[NN_index][test_movieID])):
                    KNN_ratings.append(movie_matrix.loc[NN_index][test_movieID])
            else:
                break
        unknown_user_rating = sum(KNN_ratings) / len(KNN_ratings)
        results.append(unknown_user_rating)
    else:
        results.append(0)


print(results)
output_file = str('test'+str(random.randint(1,1001))+'.dat')
file = open(output_file,'w')
for i in range(0,len(results)):
    file.write(str(results[i]) + '\n')
print('Finished! Results in ➡️',output_file)









# source: https://towardsdatascience.com/how-to-build-a-simple-recommender-system-in-python-375093c3fb7d

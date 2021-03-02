import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


data_dir = '...\data\LDOS'
files = ['/ratings.txt']
sep = '\t'
filename = data_dir + files[0]

dtypes = {
    'u_nodes': np.int32, 'v_nodes': np.str,
    'rating': np.int32}

Ratings = pd.read_csv(
    filename, sep=sep, header=None,
    names=['userId', 'movieId', 'rating'])

Mean = Ratings.groupby(by="userId",as_index=False)['rating'].mean()
Rating_avg = pd.merge(Ratings,Mean,on='userId')
Rating_avg['adg_rating']=Rating_avg['rating_x']-Rating_avg['rating_y']
Rating_avg.head()


check = pd.pivot_table(Rating_avg,values='rating_x',index='userId',columns='movieId')
check.head()

final = pd.pivot_table(Rating_avg,values='adg_rating',index='userId',columns='movieId')
final.head()

# Replacing NaN by Movie Average
final_movie = final.fillna(final.mean(axis=0))

# Replacing NaN by user Average
final_user = final.apply(lambda row: row.fillna(row.mean()), axis=1)
#print("*************************final movie head*********************************")
final_movie.head()

# user similarity on replacing NAN by user avg
b = cosine_similarity(final_user)
np.fill_diagonal(b, 0 )
similarity_with_user = pd.DataFrame(b,index=final_user.index)
similarity_with_user.columns=final_user.index
print(f"*****************sim with user*****************")
print(similarity_with_user.head())

# user similarity on replacing NAN by item(movie) avg
cosine = cosine_similarity(final_movie)
np.fill_diagonal(cosine, 0 )
similarity_with_movie = pd.DataFrame(cosine,index=final_movie.index)
similarity_with_movie.columns=final_user.index
similarity_with_movie.head()


def find_n_neighbours(df,n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index,
          index=['top{}'.format(i) for i in range(0, n)]), axis=1)
    return df


# top 30 neighbours for each user
sim_user_30_u = find_n_neighbours(similarity_with_user,5)
sim_user_30_u.to_csv(r'...\data\LDOS\LDOSSimilarUsers.csv')
#out of these neighbours find the context wise similiar neighbours 

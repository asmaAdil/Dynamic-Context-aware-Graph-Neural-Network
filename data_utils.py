from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import pdb

# For automatic dataset downloading
from urllib.request import urlopen
from zipfile import ZipFile
import shutil
import os.path


try:
    from BytesIO import BytesIO
except ImportError:
    from io import BytesIO


def data_iterator(data, batch_size):
    """
    A simple data iterator from https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/
    :param data: list of numpy tensors that need to be randomly batched across their first dimension.
    :param batch_size: int, batch_size of data_iterator.
    Assumes same first dimension size of all numpy tensors.
    :return: iterator over batches of numpy tensors
    """
    # shuffle labels and features
    max_idx = len(data[0])
    idxs = np.arange(0, max_idx)
    np.random.shuffle(idxs)
    shuf_data = [dat[idxs] for dat in data]

    # Does not yield last remainder of size less than batch_size
    for i in range(max_idx // batch_size):
        data_batch = [dat[i * batch_size:(i + 1) * batch_size] for dat in shuf_data]
        yield data_batch


def map_data(data):
    """
    Map data to proper indices in case they are not in a continues [0, N) range

    Parameters
    ----------
    data : np.int32 arrays

    Returns
    -------
    mapped_data : np.int32 arrays
    n : length of mapped_data

    """
    uniq = list(set(data))

    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    # print(f"id_dict {id_dict}")
    data = np.array([id_dict[x] for x in data])
    # print(f"dataaaaa {data}")
    max_value = np.max(data)
    n = len(uniq)

    return data, id_dict, n


def download_dataset(dataset, files, data_dir):
    """ Downloads dataset if files are not present. """

    if not np.all([os.path.isfile(data_dir + f) for f in files]):
        url = "http://files.grouplens.org/datasets/movielens/" + dataset.replace('_', '-') + '.zip'
        request = urlopen(url)

        print('Downloading %s dataset' % dataset)

        if dataset in ['ml_100k', 'ml_1m']:
            target_dir = 'raw_data/' + dataset.replace('_', '-')
        elif dataset == 'ml_10m':
            target_dir = 'raw_data/' + 'ml-10M100K'
        else:
            raise ValueError('Invalid dataset option %s' % dataset)

        with ZipFile(BytesIO(request.read())) as zip_ref:
            zip_ref.extractall('raw_data/')

        os.rename(target_dir, data_dir)
        # shutil.rmtree(target_dir)


def load_data(fname, seed=1234, verbose=True):
    """ Loads dataset and creates adjacency matrix
    and feature matrix

    Parameters
    ----------
    fname : str, dataset
    seed: int, dataset shuffling seed
    verbose: to print out statements or not

    Returns
    -------
    num_users : int
        Number of users and items respectively

    num_items : int

    u_nodes : np.int32 arrays
        User indices

    v_nodes : np.int32 array
        item (movie) indices

    ratings : np.float32 array
        User/item ratings s.t. ratings[k] is the rating given by user u_nodes[k] to
        item v_nodes[k]. Note that that the all pairs u_nodes[k]/v_nodes[k] are unique, but
        not necessarily all u_nodes[k] or all v_nodes[k] separately.

    u_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    v_features: np.float32 array, or None
        If present in dataset, contains the features of the users.

    seed: int,
        For datashuffling seed with pythons own random.shuffle, as in CF-NADE.

    """

    u_features = None
    v_features = None
    edge_features = None
    sim_users=None

    print('Loading dataset', fname)

    data_dir = 'raw_data/' + fname
    if fname == 'DePaul':
        print("I am DePaul")
        data_dir = '...\data\DePaul'
        files = ['/ratings.txt']
        sep = '\t'

        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.str,
            'ratings': np.float32}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'time', 'location', 'companion'], dtype=dtypes)

        data_array = data.values.tolist()
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}  # {0:1, 1:2, 2:3, 3:4, 4:5}

        time = data_array[:, 3]
        location = data_array[:, 4]
        companion = data_array[:, 5]

        time_s = set(list(time))
        time_s.remove('-1')
        time_dict = {f: i for i, f in enumerate(time_s, start=0)}

        location_s = set(list(location))
        location_s.remove('-1')
        location_dict = {f: i for i, f in enumerate(location_s, start=len(time_dict))}

        companion_s = set(list(companion))
        companion_s.remove('-1')
        companion_dict = {f: i for i, f in enumerate(companion_s, start=4)}


        N_edgeFeature = len(time_dict) + len(location_dict) + len(companion_dict)
        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
        edge_features = np.zeros((num_users, num_items, N_edgeFeature), dtype=np.float32)  # UxIxC
        edge_features_list = np.zeros((len(u_nodes_ratings), N_edgeFeature), dtype=np.float32)
        edge_features_list_f = []

        j = 0
        i=0
        for _, row in data.iterrows():
            u_id = row['u_nodes']
            v_id = row['v_nodes']
            if v_id in v_dict.keys() and u_id in u_dict.keys():
                if row['time'] != '-1':
                    edge_features[u_dict[u_id], v_dict[v_id], time_dict[row['time']]] = 1.
                    edge_features_list[i, time_dict[row['time']]] = 1.
                if row['location'] != '-1':
                    edge_features[u_dict[u_id], v_dict[v_id], location_dict[row['location']]] = 1.
                    edge_features_list[i,  location_dict[row['location']]]= 1.
                if row['companion'] != '-1':
                    edge_features[u_dict[u_id], v_dict[v_id], companion_dict[row['companion']]] = 1.
                    edge_features_list[i, companion_dict[row['companion']]] = 1.

                temp1=edge_features_list[i,:].tolist()
                edge_features_list_f.append(temp1)
                j += 1
                i += 1
        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)
        ind = np.array(np.where(edge_features != 0)).T  # (u v c) #non zero indices
        u_features = None
        v_features = None
        sim_users={}

    elif fname == 'Trip':
        print("I am Trip Advisor")
        data_dir = '...\data\TripAdvisor'
        files = ['/TA-ratingContext.txt', '/TA-IFeatures.txt', '/TA-UFeatures.txt','/TripSimilarUsers.txt']  #,
        sep = '\t'

        filename = data_dir + files[0]
        print(f"filename {filename}")
        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'company'], dtype=dtypes)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        # random.seed(seed)
        # random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}  # {0:1, 1:2, 2:3, 3:4, 4:5}

        company = data_array[:, 3]
        company_s = set(list(company))
        company_dict = {f: i for i, f in enumerate(company_s, start=0)}

        N_edgeFeature = len(company_dict)
        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
        edge_features = np.zeros((num_users, num_items, N_edgeFeature), dtype=np.float32)  # UxIxC
        edge_features_list=np.zeros((len(u_nodes_ratings), N_edgeFeature),dtype=np.float32)
        edge_features_list_f=[]

        i=0
        j=0
        for _, row in data.iterrows():
            u_id = row['u_nodes']
            v_id = row['v_nodes']
            if v_id in v_dict.keys() and u_id in u_dict.keys():
                if row['company'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], company_dict[row['company']]] = 1.
                    edge_features_list[i,company_dict[row['company']] ]=1.


                temp1=edge_features_list[i,:].tolist()
                edge_features_list_f.append(temp1)
                #print (temp1)
                i=i+1
                j += 1


        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)
        ##############################################################################################################
        filename = data_dir + files[3]
        print(f"filename {filename}")

        dtypes = {
            'u_nodes': np.int32, 'v1_nodes': np.int32,'v2_nodes': np.int32,'v3_nodes': np.int32,'v4_nodes': np.int32,'v5_nodes': np.int32,}

        data_sim_user = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v1_nodes', 'v2_nodes', 'v3_nodes', 'v4_nodes', 'v5_nodes'], dtype=dtypes)
        sim_users={}
        for _, row in data_sim_user.iterrows():
            u_id = row['u_nodes']
            if u_id in u_dict.keys():
                key=u_dict[u_id]
                sim_users.setdefault(key, [])
                sim_users[key].append(u_dict[row['v1_nodes']])
                sim_users[key].append(u_dict[row['v2_nodes']])
                sim_users[key].append(u_dict[row['v3_nodes']])
                sim_users[key].append(u_dict[row['v4_nodes']])
                sim_users[key].append(u_dict[row['v5_nodes']])
        ##############################################################################################################
        # Movie features (genres)
        sep = '\t'
        movie_file = data_dir + files[1]
        movie_headers = ['itemID', 'IC', 'IS', 'ITZ']

        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        IC_list = movie_df['IC'].values.tolist()
        IS_list = movie_df['IS'].values.tolist()
        ITZ_list = movie_df['ITZ'].values.tolist()

        # movieCountry_list= movieCountry.values.tolist()  #movieLanguage_list= movieLanguage.values.tolist()    #movieYear_list=  movieYear.values.tolist()
        IC_set = set(IC_list)
        #IC_set.discard(0)
        # movieCountry_set= set(movieCountry_list)  #movieCountry_set.discard(-1)   #movieLanguage_set=set(movieLanguage_list) #movieLanguage_set.discard(-1) #movieYear_set=set(movieYear_list) #movieYear_.discard(-1)

        IS_set = set(IS_list)
        #IS_set.discard(0)

        ITZ_set = set(ITZ_list)
        #ITZ_set.discard(0)

        IC_dict = {f: i for i, f in enumerate(IC_set, start=0)}
        IS_dict = {f: i for i, f in enumerate(IS_set, start=(len(IC_dict)))}
        ITZ_dict = {f: i for i, f in enumerate(ITZ_set, start=(len(IC_dict)+len(IS_dict)))}

        n_features = len(IC_dict)+len(IS_dict)+len(ITZ_dict)
        v_features = np.zeros((num_items, n_features), dtype=np.float32)

        for _, row in movie_df.iterrows():
            v_id = row['itemID']
            if v_id in v_dict.keys():
                v_features[v_dict[v_id], IC_dict[row['IC']]] = 1.
                v_features[v_dict[v_id], IS_dict[row['IS']]] = 1.
                v_features[v_dict[v_id], ITZ_dict[row['ITZ']]] = 1.

        # User feature
        sep = '\t'
        users_file = data_dir + files[2]
        users_headers = ['user id', 'UC', 'US']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')
        UC_set = set(users_df['UC'].values.tolist())
        UC_dict = {f: i for i, f in enumerate(UC_set, start=0)}
        US_set = set(users_df['US'].values.tolist())
        US_dict = {f: i for i, f in enumerate(US_set, start=len(UC_dict))}
        num_feats =  len(UC_dict) + len(US_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # gender
                u_features[u_dict[u_id], UC_dict[row['UC']]] = 1
                # country
                u_features[u_dict[u_id], US_dict[row['US']]] = 1

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)
        # edge_features= sp.csr_matrix(edge_features)

        ind = np.array(np.where(edge_features != 0)).T  # (u v c) #non zero indices
        print("LOAD DATA DONE")
    elif fname == 'Travel_STS':
        print("I am Travel_STS")
        data_dir = '...\data\Travel_STS'
        files = ['/Travel-User-Item-Rating-Context.txt', '/Travel-User-Info.txt']
        sep = '\t'

        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'distance', 'time_available', 'temperature', 'crowdedness',
                   'knowledge_of_surroundings', 'season', 'budget', 'daytime', 'weather', 'companion', 'mood',
                   'weekday', 'travelgoal', 'means_of_transport', 'category'], dtype=dtypes)

        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])
        rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}  # {0:1, 1:2, 2:3, 3:4, 4:5}

        distance = data_array[:, 3]
        time_available = data_array[:, 4]
        temperature = data_array[:, 5]
        crowdedness = data_array[:, 6]
        knowledge_of_surroundings = data_array[:, 7]
        season = data_array[:, 8]
        budget = data_array[:, 9]
        daytime = data_array[:, 10]
        weather = data_array[:, 11]
        companion = data_array[:, 12]
        mood = data_array[:, 13]
        weekday = data_array[:, 14]
        travelgoal = data_array[:, 15]
        means_of_transport = data_array[:, 16]
        category = data_array[:, 17]

        distance_s = set(list(distance))
        distance_s.discard(-1)
        distance_dict = {f: i for i, f in enumerate(distance_s, start=0)}

        time_available_s = set(list(time_available))
        time_available_s.discard(-1)
        time_available_dict = {f: i for i, f in enumerate(time_available_s, start=len(distance_dict))}

        temperature_s = set(list(temperature))
        temperature_s.discard(-1)
        temperature_dict = {f: i for i, f in
                            enumerate(temperature_s, start=len(distance_dict) + len(time_available_dict))}

        crowdedness_s = set(list(crowdedness))
        crowdedness_s.discard(-1)
        crowdedness_dict = {f: i for i, f in enumerate(crowdedness_s,
                                                       start=len(distance_dict) + len(time_available_dict) + len(
                                                           temperature_dict))}

        knowledge_of_surroundings_s = set(list(knowledge_of_surroundings))
        knowledge_of_surroundings_s.discard(-1)
        knowledge_of_surroundings_dict = {f: i for i, f in enumerate(knowledge_of_surroundings_s,
                                                                     start=len(distance_dict) + len(
                                                                         time_available_dict) + len(
                                                                         temperature_dict) + len(crowdedness_dict))}

        season_s = set(list(season))
        season_s.discard(-1)
        season_dict = {f: i for i, f in enumerate(season_s, start=len(distance_dict) + len(time_available_dict) + len(
            temperature_dict) + len(crowdedness_dict) + len(knowledge_of_surroundings_dict))}

        budget_s = set(list(budget))
        budget_s.discard(-1)
        budget_dict = {f: i for i, f in enumerate(budget_s, start=len(distance_dict) + len(time_available_dict) + len(
            temperature_dict) + len(crowdedness_dict) + len(knowledge_of_surroundings_dict) + len(season_dict))}

        daytime_s = set(list(daytime))
        daytime_s.discard(-1)
        daytime_dict = {f: i for i, f in enumerate(daytime_s,
                                                   start=len(distance_dict) + len(time_available_dict) + len(
                                                       temperature_dict) + len(crowdedness_dict) + len(
                                                       knowledge_of_surroundings_dict) + len(season_dict) + len(
                                                       budget_dict))}

        weather_s = set(list(weather))
        weather_s.discard(-1)
        weather_dict = {f: i for i, f in enumerate(weather_s,
                                                   start=len(distance_dict) + len(time_available_dict) + len(
                                                       temperature_dict) + len(crowdedness_dict) + len(
                                                       knowledge_of_surroundings_dict) + len(season_dict) + len(
                                                       budget_dict) + len(daytime_dict))}

        companion_s = set(list(companion))
        companion_s.discard(-1)
        companion_dict = {f: i for i, f in enumerate(companion_s,
                                                     start=len(distance_dict) + len(time_available_dict) + len(
                                                         temperature_dict) + len(crowdedness_dict) + len(
                                                         knowledge_of_surroundings_dict) + len(season_dict) + len(
                                                         budget_dict) + len(daytime_dict) + len(weather_dict))}

        mood_s = set(list(mood))
        mood_s.discard(-1)
        mood_dict = {f: i for i, f in enumerate(mood_s,
                                                start=len(distance_dict) + len(time_available_dict) + len(
                                                    temperature_dict) + len(crowdedness_dict) + len(
                                                    knowledge_of_surroundings_dict) + len(season_dict) + len(
                                                    budget_dict) + len(daytime_dict) + len(weather_dict) + len(
                                                    companion_dict))}

        weekday_s = set(list(weekday))
        weekday_s.discard(-1)
        weekday_dict = {f: i for i, f in enumerate(weekday_s,
                                                   start=len(distance_dict) + len(time_available_dict) + len(
                                                       temperature_dict) + len(crowdedness_dict) + len(
                                                       knowledge_of_surroundings_dict) + len(season_dict) + len(
                                                       budget_dict) + len(daytime_dict) + len(weather_dict) + len(
                                                       companion_dict) + len(mood_dict))}

        travelgoal_s = set(list(travelgoal))
        travelgoal_s.discard(-1)
        travelgoal_dict = {f: i for i, f in enumerate(travelgoal_s,
                                                      start=len(distance_dict) + len(time_available_dict) + len(
                                                          temperature_dict) + len(crowdedness_dict) + len(
                                                          knowledge_of_surroundings_dict) + len(season_dict) + len(
                                                          budget_dict) + len(daytime_dict) + len(weather_dict) + len(
                                                          companion_dict) + len(mood_dict) + len(weekday_dict))}

        means_of_transport_s = set(list(means_of_transport))
        means_of_transport_s.discard(-1)
        means_of_transport_dict = {f: i for i, f in enumerate(means_of_transport_s,
                                                              start=len(distance_dict) + len(time_available_dict) + len(
                                                                  temperature_dict) + len(crowdedness_dict) + len(
                                                                  knowledge_of_surroundings_dict) + len(
                                                                  season_dict) + len(
                                                                  budget_dict) + len(daytime_dict) + len(
                                                                  weather_dict) + len(companion_dict) + len(
                                                                  mood_dict) + len(weekday_dict) + len(
                                                                  travelgoal_dict))}

        category_s = set(list(category))
        category_s.discard(-1)
        category_dict = {f: i for i, f in enumerate(category_s,
                                                    start=len(distance_dict) + len(time_available_dict) + len(
                                                        temperature_dict) + len(crowdedness_dict) + len(
                                                        knowledge_of_surroundings_dict) + len(
                                                        season_dict) + len(
                                                        budget_dict) + len(daytime_dict) + len(
                                                        weather_dict) + len(companion_dict) + len(
                                                        mood_dict) + len(weekday_dict) + len(
                                                        travelgoal_dict))}

        N_edgeFeature = len(distance_dict) + len(time_available_dict) + len(
            temperature_dict) + len(crowdedness_dict) + len(
            knowledge_of_surroundings_dict) + len(season_dict) + len(
            budget_dict) + len(daytime_dict) + len(weather_dict) + len(companion_dict) + len(mood_dict) + len(
            weekday_dict) \
                        + len(travelgoal_dict) + len(means_of_transport_dict) + +len(category_dict)

        print(f"N_edgeFeature {N_edgeFeature}")
        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
        edge_features = np.zeros((num_users, num_items, N_edgeFeature), dtype=np.float32)  # UxIxC
        edge_features_list=np.zeros((len(u_nodes_ratings), N_edgeFeature),dtype=np.float32)
        edge_features_list_f=[]


        j = 0
        i=0
        for _, row in data.iterrows():
            u_id = row['u_nodes']
            v_id = row['v_nodes']
            if v_id in v_dict.keys() and u_id in u_dict.keys():
                if row['distance'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], distance_dict[row['distance']]] = 1.
                    edge_features_list[i,distance_dict[row['distance']]] = 1.

                if row['time_available'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], time_available_dict[row['time_available']]] = 1.
                    edge_features_list[i,  time_available_dict[row['time_available']]] = 1.

                if row['temperature'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], temperature_dict[row['temperature']]] = 1.
                    edge_features_list[i, temperature_dict[row['temperature']]] = 1.

                if row['crowdedness'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], crowdedness_dict[row['crowdedness']]] = 1.
                    edge_features_list[i, crowdedness_dict[row['crowdedness']]] = 1.

                if row['knowledge_of_surroundings'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], knowledge_of_surroundings_dict[row['knowledge_of_surroundings']]] = 1.
                    edge_features_list[i, knowledge_of_surroundings_dict[row['knowledge_of_surroundings']]] = 1.

                if row['season'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], season_dict[row['season']]] = 1.
                    edge_features_list[i, season_dict[row['season']]] = 1.

                if row['budget'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], budget_dict[row['budget']]] = 1.
                    edge_features_list[i,budget_dict[row['budget']]] = 1.

                if row['daytime'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], daytime_dict[row['daytime']]] = 1.
                    edge_features_list[i, daytime_dict[row['daytime']]] = 1.

                if row['weather'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], weather_dict[row['weather']]] = 1.
                    edge_features_list[i, weather_dict[row['weather']]] = 1.

                if row['companion'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], companion_dict[row['companion']]] = 1.
                    edge_features_list[i,  companion_dict[row['companion']]] = 1.

                if row['mood'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], mood_dict[row['mood']]] = 1.
                    edge_features_list[i,   mood_dict[row['mood']]] = 1.

                if row['weekday'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], weekday_dict[row['weekday']]] = 1.
                    edge_features_list[i,  weekday_dict[row['weekday']]] = 1.

                if row['travelgoal'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], travelgoal_dict[row['travelgoal']]] = 1.
                    edge_features_list[i,   travelgoal_dict[row['travelgoal']]] = 1.

                if row['means_of_transport'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], means_of_transport_dict[row['means_of_transport']]] = 1.
                    edge_features_list[i,  means_of_transport_dict[row['means_of_transport']]] = 1.

                if row['category'] != -1:
                    edge_features[u_dict[u_id], v_dict[v_id], category_dict[row['category']]] = 1.
                    edge_features_list[i,  category_dict[row['category']]] = 1.

                temp1 = edge_features_list[i, :].tolist()
                edge_features_list_f.append(temp1)
                j += 1
                i +=1

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)

        ##############################################################################################################
        # item features (genres)= None
        #############################################################################################################

        # User feature
        sep = '\t'
        users_file = data_dir + files[1]
        users_headers = ['user_id', 'age', 'sex', 'opennessToExperience', 'conscientiousness', 'extraversion',
                         'agreeableness', 'emotionalStability']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')
        sex_dict = {'m': 0., 'f': 1.}

        age = users_df['age'].values
        age_max = age.max()

        opennessToExperience_set = set(users_df['opennessToExperience'].values.tolist())
        opennessToExperience_set.discard(-1)
        opennessToExperience_dict = {f: i for i, f in enumerate(opennessToExperience_set, start=2)}

        conscientiousness_set = set(users_df['conscientiousness'].values.tolist())
        conscientiousness_set.discard(-1)
        conscientiousness_dict = {f: i for i, f in
                                  enumerate(conscientiousness_set, start=2 + len(opennessToExperience_dict))}

        extraversion_set = set(users_df['extraversion'].values.tolist())
        extraversion_set.discard(-1)
        extraversion_dict = {f: i for i, f in enumerate(extraversion_set,
                                                        start=2 + len(opennessToExperience_dict) + len(
                                                            conscientiousness_dict))}

        agreeableness_set = set(users_df['agreeableness'].values.tolist())
        agreeableness_set.discard(-1)
        agreeableness_dict = {f: i for i, f in enumerate(agreeableness_set,
                                                         start=2 + len(opennessToExperience_dict) + len(
                                                             conscientiousness_dict) + len(extraversion_dict))}

        emotionalStability_set = set(users_df['emotionalStability'].values.tolist())
        emotionalStability_set.discard(-1)
        emotionalStability_dict = {f: i for i, f in enumerate(emotionalStability_set,
                                                              start=2 + len(opennessToExperience_dict) + len(
                                                                  conscientiousness_dict) + len(
                                                                  extraversion_dict) + len(agreeableness_dict))}

        num_feats = 2 + len(opennessToExperience_dict) + len(conscientiousness_dict) + len(agreeableness_dict) + len(
            emotionalStability_dict) + len(extraversion_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                # age np.float(age_max)
                if row['age'] != -1: u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                if row['sex'] != '-1': u_features[u_dict[u_id], 1] = sex_dict[row['sex']]
                if row['opennessToExperience'] != -1: u_features[
                    u_dict[u_id], opennessToExperience_dict[row['opennessToExperience']]] = 1.
                if row['conscientiousness'] != -1: u_features[
                    u_dict[u_id], conscientiousness_dict[row['conscientiousness']]] = 1.
                if row['emotionalStability'] != -1: u_features[
                    u_dict[u_id], emotionalStability_dict[row['emotionalStability']]] = 1.
                if row['agreeableness'] != -1: u_features[u_dict[u_id], agreeableness_dict[row['agreeableness']]] = 1.
                if row['extraversion'] != -1: u_features[u_dict[u_id], extraversion_dict[row['extraversion']]] = 1.

        u_features = sp.csr_matrix(u_features)
        # edge_features= sp.csr_matrix(edge_features)

        ind = np.array(np.where(edge_features != 0)).T  # (u v c) #non zero indices
    elif fname == 'LDOS':
        print("I am LDOS")
        data_dir = '...\data\LDOS'
        files = ['/LDOS-User-Item-RatingContext.txt', '/LDOS-ItemFeatures.txt', '/User-info.txt', '/LDOSSimilarUsers.txt']
        sep = '\t'

        filename = data_dir + files[0]
        print(f"filename {filename}")
        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32}

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'time', 'daytype', 'season', 'location', 'weather', 'social',
                   'endEmo', 'dominantEmo', 'mood', 'physical', 'decision', 'interaction'], dtype=dtypes)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        # random.seed(seed)
        # random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        rating_dict = {r: i for i, r in enumerate(np.sort(np.unique(ratings)).tolist())}  # {0:1, 1:2, 2:3, 3:4, 4:5}
        print(f"rating_dict {rating_dict}")

        time = data_array[:, 3]
        daytype = data_array[:, 4]
        season = data_array[:, 5]
        location = data_array[:, 6]
        weather = data_array[:, 7]
        social = data_array[:, 8]
        endEmo = data_array[:, 9]
        dominantEmo = data_array[:, 10]
        mood = data_array[:, 11]
        physical = data_array[:, 12]
        decision = data_array[:, 13]
        interaction = data_array[:, 14]

        time_s = set(list(time))
        time_s.discard(0)
        time_dict = {f: i for i, f in enumerate(time_s, start=0)}

        daytype_s = set(list(daytype))
        daytype_s.discard(0)
        daytype_dict = {f: i for i, f in enumerate(daytype_s, start=4)}

        season_s = set(list(season))
        season_s.discard(0)
        season_dict = {f: i for i, f in enumerate(season_s, start=7)}

        location_s = set(list(location))
        location_s.discard(0)
        location_dict = {f: i for i, f in enumerate(location_s, start=11)}

        weather_s = set(list(weather))
        weather_s.discard(0)
        weather_dict = {f: i for i, f in enumerate(weather_s, start=14)}

        social_s = set(list(social))
        social_s.discard(0)
        social_dict = {f: i for i, f in enumerate(social_s, start=19)}

        endEmo_s = set(list(endEmo))
        endEmo_s.discard(0)
        endEmo_dict = {f: i for i, f in enumerate(endEmo_s, start=26)}

        dominantEmo_s = set(list(dominantEmo))
        dominantEmo_s.discard(0)
        dominantEmo_dict = {f: i for i, f in enumerate(dominantEmo_s, start=33)}

        mood_s = set(list(mood))
        mood_s.discard(0)
        mood_dict = {f: i for i, f in enumerate(mood_s, start=40)}

        physical_s = set(list(physical))
        physical_s.discard(0)
        physical_dict = {f: i for i, f in enumerate(physical_s, start=43)}

        decision_s = set(list(decision))
        decision_s.discard(0)
        decision_dict = {f: i for i, f in enumerate(decision_s, start=45)}

        interaction_s = set(list(interaction))
        interaction_s.discard(0)
        interaction_dict = {f: i for i, f in enumerate(interaction_s, start=47)}

        N_edgeFeature = len(time_dict) + len(daytype_dict) + len(season_dict) + len(location_dict) + len(
            weather_dict) + len(social_dict) + len(endEmo_dict) + len(dominantEmo_dict) + len(mood_dict) + len(
            physical_dict) + len(decision_dict) + len(interaction_dict)
        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)
        edge_features = np.zeros((num_users, num_items, N_edgeFeature), dtype=np.float32)  # UxIxC
        edge_features_list=np.zeros((2296, N_edgeFeature),dtype=np.float32)
        edge_features_list_f=[]


        i=0
        j=0
        for _, row in data.iterrows():
            u_id = row['u_nodes']
            v_id = row['v_nodes']
            if v_id in v_dict.keys() and u_id in u_dict.keys():
                if row['time'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], time_dict[row['time']]] = 1.
                    edge_features_list[i,time_dict[row['time']] ]=1.

                if row['daytype'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], daytype_dict[row['daytype']]] = 1.
                    edge_features_list [i, daytype_dict[row['daytype']]] = 1.
                if row['season'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], season_dict[row['season']]] = 1.
                    edge_features_list [i,season_dict[row['season']]] = 1.
                if row['location'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], location_dict[row['location']]] = 1.
                    edge_features_list [i, location_dict[row['location']]] = 1.
                if row['weather'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], weather_dict[row['weather']]] = 1.
                    edge_features_list [i, weather_dict[row['weather']]] = 1.
                if row['social'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], social_dict[row['social']]] = 1.
                    edge_features_list [i, social_dict[row['social']]] = 1.
                if row['endEmo'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], endEmo_dict[row['endEmo']]] = 1.
                    edge_features_list [i, endEmo_dict[row['endEmo']]] = 1.
                if row['dominantEmo'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], dominantEmo_dict[row['dominantEmo']]] = 1.
                    edge_features_list [i, dominantEmo_dict[row['dominantEmo']]] = 1.
                if row['mood'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], mood_dict[row['mood']]] = 1.
                    edge_features_list  [i, mood_dict[row['mood']]] = 1.
                if row['physical'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], physical_dict[row['physical']]] = 1.
                    edge_features_list  [i, physical_dict[row['physical']]] = 1.
                if row['interaction'] != 0:
                    edge_features[u_dict[u_id], v_dict[v_id], interaction_dict[row['interaction']]] = 1.
                    edge_features_list  [i, interaction_dict[row['interaction']]] = 1.


                temp1=edge_features_list[i,:].tolist()
                edge_features_list_f.append(temp1)
                #print (temp1)
                i=i+1
                j += 1


        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)
        ##############################################################################################################
        filename = data_dir + files[3]
        print(f"filename {filename}")

        dtypes = {
            'u_nodes': np.int32, 'v1_nodes': np.int32,'v2_nodes': np.int32,'v3_nodes': np.int32,'v4_nodes': np.int32,'v5_nodes': np.int32,}

        data_sim_user = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v1_nodes', 'v2_nodes', 'v3_nodes', 'v4_nodes', 'v5_nodes'], dtype=dtypes)
        sim_users={}
        for _, row in data_sim_user.iterrows():
            u_id = row['u_nodes']
            if u_id in u_dict.keys():
                key=u_dict[u_id]
                sim_users.setdefault(key, [])
                sim_users[key].append(u_dict[row['v1_nodes']])
                sim_users[key].append(u_dict[row['v2_nodes']])
                sim_users[key].append(u_dict[row['v3_nodes']])
                sim_users[key].append(u_dict[row['v4_nodes']])
                sim_users[key].append(u_dict[row['v5_nodes']])
        ##############################################################################################################
        # Movie features (genres)
        sep = '\t'
        movie_file = data_dir + files[1]
        movie_headers = ['itemID', 'director', 'movieCountry', 'movieLanguage',
                         'movieYear', 'genre1', 'genre2', 'genre3', 'actor1',
                         'actor2', 'actor3', 'budget', 'ImdbUrl']

        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        director_list = movie_df['director'].values.tolist()
        genre_list = movie_df['genre1'].values.tolist() + movie_df['genre2'].values.tolist() + movie_df[
            'genre3'].values.tolist()
        actor_list = movie_df['actor1'].values.tolist() + movie_df['actor2'].values.tolist() + movie_df[
            'actor3'].values.tolist()

        # movieCountry_list= movieCountry.values.tolist()  #movieLanguage_list= movieLanguage.values.tolist()    #movieYear_list=  movieYear.values.tolist()
        director_set = set(director_list)
        director_set.discard(0)
        # movieCountry_set= set(movieCountry_list)  #movieCountry_set.discard(-1)   #movieLanguage_set=set(movieLanguage_list) #movieLanguage_set.discard(-1) #movieYear_set=set(movieYear_list) #movieYear_.discard(-1)

        genre_set = set(genre_list)
        genre_set.discard(0)

        actor_set = set(actor_list)
        actor_set.discard(0)

        director_dict = {f: i for i, f in enumerate(director_set, start=0)}
        genre_dict = {f: i for i, f in enumerate(genre_set, start=814)}
        actor_dict = {f: i for i, f in enumerate(actor_set, start=837)}

        budget = movie_df['budget'].values
        max_budget = budget.max()

        n_features = 1 + len(director_dict) + len(genre_dict) + len(actor_dict)
        v_features = np.zeros((num_items, n_features), dtype=np.float32)

        for _, row in movie_df.iterrows():
            v_id = row['itemID']
            if v_id in v_dict.keys():
                if row['director'] != 0: v_features[v_dict[v_id], director_dict[row['director']]] = 1.
                if row['genre1'] != 0: v_features[v_dict[v_id], genre_dict[row['genre1']]] = 1.
                if row['genre2'] != 0: v_features[v_dict[v_id], genre_dict[row['genre2']]] = 1.
                if row['genre3'] != 0: v_features[v_dict[v_id], genre_dict[row['genre3']]] = 1.
                if row['actor1'] != 0: v_features[v_dict[v_id], actor_dict[row['actor1']]] = 1.
                if row['actor2'] != 0: v_features[v_dict[v_id], actor_dict[row['actor2']]] = 1.
                if row['actor3'] != 0: v_features[v_dict[v_id], actor_dict[row['actor3']]] = 1.
                v_features[v_dict[v_id], n_features - 1] = row['budget'] / max_budget


        # User feature
        sep = '\t'
        users_file = data_dir + files[2]
        users_headers = ['user id', 'age', 'sex', 'city', 'country']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')
        gender_dict = {1: 0., 2: 1.}
        country_set = set(users_df['country'].values.tolist())
        country_set.discard(-1)
        country_dict = {f: i for i, f in enumerate(country_set, start=2)}
        num_feats = 2 + len(country_dict)
        age = users_df['age'].values
        age_max = age.max()

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age np.float(age_max)
                u_features[u_dict[u_id], 0] = row['age'] / np.float(age_max)
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['sex']]
                # country
                u_features[u_dict[u_id], country_dict[row['country']]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)
        # edge_features= sp.csr_matrix(edge_features)

        ind = np.array(np.where(edge_features != 0)).T  # (u v c) #non zero indices

    elif fname == 'ml_100k':

        # Check if files exist and download otherwise
        files = ['/u.data', '/u.item', '/u.user']

        download_dataset(fname, files, data_dir)

        sep = '\t'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int32, 'v_nodes': np.int32,
            'ratings': np.float32, 'timestamp': np.float64}
        # userID   itemID rating time daytype season location weather social endEmo dominantEmo mood physical decision interaction

        data = pd.read_csv(
            filename, sep=sep, header=None,
            names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], dtype=dtypes)

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        print(f"u_nodes_ratings {u_nodes_ratings} v_nodes_ratings {v_nodes_ratings} ratings  {ratings}")

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        print(f"u_nodes_ratings  {u_nodes_ratings} v_nodes_ratings {v_nodes_ratings} ratings{ratings}")

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int32)
        ratings = ratings.astype(np.float64)

        # Movie features (genres)
        sep = r'|'
        movie_file = data_dir + files[1]
        movie_headers = ['movie id', 'movie title', 'release date', 'video release date',
                         'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                         'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                         'Thriller', 'War', 'Western']
        movie_df = pd.read_csv(movie_file, sep=sep, header=None,
                               names=movie_headers, engine='python')

        genre_headers = movie_df.columns.values[6:]
        num_genres = genre_headers.shape[0]

        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, g_vec in zip(movie_df['movie id'].values.tolist(), movie_df[genre_headers].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                v_features[v_dict[movie_id], :] = g_vec

        # User features

        sep = r'|'
        users_file = data_dir + files[2]
        users_headers = ['user id', 'age', 'gender', 'occupation', 'zip code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        occupation = set(users_df['occupation'].values.tolist())

        gender_dict = {'M': 0., 'F': 1.}
        occupation_dict = {f: i for i, f in enumerate(occupation, start=2)}

        num_feats = 2 + len(occupation_dict)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user id']
            if u_id in u_dict.keys():
                # age
                u_features[u_dict[u_id], 0] = row['age']
                # gender
                u_features[u_dict[u_id], 1] = gender_dict[row['gender']]
                # occupation
                u_features[u_dict[u_id], occupation_dict[row['occupation']]] = 1.

        u_features = sp.csr_matrix(u_features)

    elif fname == 'ml_1m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat', '/movies.dat', '/users.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'
        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)

        # Load movie features
        movies_file = data_dir + files[1]

        movies_headers = ['movie_id', 'title', 'genre']
        movies_df = pd.read_csv(movies_file, sep=sep, header=None,
                                names=movies_headers, engine='python')

        # Extracting all genres
        genres = []
        for s in movies_df['genre'].values:
            genres.extend(s.split('|'))

        genres = list(set(genres))
        num_genres = len(genres)

        genres_dict = {g: idx for idx, g in enumerate(genres)}

        # Creating 0 or 1 valued features for all genres
        v_features = np.zeros((num_items, num_genres), dtype=np.float32)
        for movie_id, s in zip(movies_df['movie_id'].values.tolist(), movies_df['genre'].values.tolist()):
            # Check if movie_id was listed in ratings file and therefore in mapping dictionary
            if movie_id in v_dict.keys():
                gen = s.split('|')
                for g in gen:
                    v_features[v_dict[movie_id], genres_dict[g]] = 1.

        # Load user features
        users_file = data_dir + files[2]
        users_headers = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
        users_df = pd.read_csv(users_file, sep=sep, header=None,
                               names=users_headers, engine='python')

        # Extracting all features
        cols = users_df.columns.values[1:]

        cntr = 0
        feat_dicts = []
        for header in cols:
            d = dict()
            feats = np.unique(users_df[header].values).tolist()
            d.update({f: i for i, f in enumerate(feats, start=cntr)})
            feat_dicts.append(d)
            cntr += len(d)

        num_feats = sum(len(d) for d in feat_dicts)

        u_features = np.zeros((num_users, num_feats), dtype=np.float32)
        for _, row in users_df.iterrows():
            u_id = row['user_id']
            if u_id in u_dict.keys():
                for k, header in enumerate(cols):
                    u_features[u_dict[u_id], feat_dicts[k][row[header]]] = 1.

        u_features = sp.csr_matrix(u_features)
        v_features = sp.csr_matrix(v_features)

    elif fname == 'ml_10m':

        # Check if files exist and download otherwise
        files = ['/ratings.dat']
        download_dataset(fname, files, data_dir)

        sep = r'\:\:'

        filename = data_dir + files[0]

        dtypes = {
            'u_nodes': np.int64, 'v_nodes': np.int64,
            'ratings': np.float32, 'timestamp': np.float64}

        # use engine='python' to ignore warning about switching to python backend when using regexp for sep
        data = pd.read_csv(filename, sep=sep, header=None,
                           names=['u_nodes', 'v_nodes', 'ratings', 'timestamp'], converters=dtypes, engine='python')

        # shuffle here like cf-nade paper with python's own random class
        # make sure to convert to list, otherwise random.shuffle acts weird on it without a warning
        data_array = data.values.tolist()
        random.seed(seed)
        random.shuffle(data_array)
        data_array = np.array(data_array)

        u_nodes_ratings = data_array[:, 0].astype(dtypes['u_nodes'])
        v_nodes_ratings = data_array[:, 1].astype(dtypes['v_nodes'])
        ratings = data_array[:, 2].astype(dtypes['ratings'])

        u_nodes_ratings, u_dict, num_users = map_data(u_nodes_ratings)
        v_nodes_ratings, v_dict, num_items = map_data(v_nodes_ratings)

        u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(np.int64), v_nodes_ratings.astype(np.int64)
        ratings = ratings.astype(np.float32)


    else:
        raise ValueError('Dataset name not recognized: ' + fname)

    if verbose:

        print("**********************************************************************")
        print(f"Load Data  data_utils {fname}  ")
        print('Number of users = %d' % num_users)
        print('Number of items = %d' % num_items)
        print('Number of links = %d' % ratings.shape[0])
        print(f"ratings     {ratings.shape}")
        print(f"u_nodes_indicex     {u_nodes_ratings.shape}")
        print(f"v_nodes_indices     {v_nodes_ratings.shape}")
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_users * num_items)))

    return num_users, num_items, u_nodes_ratings, v_nodes_ratings, ratings, u_features, v_features, edge_features, edge_features_list_f,sim_users, rating_dict




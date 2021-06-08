import os
import argparse
import configparser
import pandas as pd
import numpy as np



def configuration():
    parser = argparse.ArgumentParser(description='Movie recommendation system')
    parser.add_argument('-u', '--userId',action  = "store",type = int,default = 100,help='User Id')
    return parser


def check_paths(source  = None, destination = None, file_name = None):
    
    if source is not None:
        if isinstance(source, list):
            for path in source:
                if not os.path.exists(path):
                    raise RuntimeError(f"Could not find source path '{path}'. Please check source path")
        else:
            if not os.path.exists(source):
                    raise RuntimeError(f"Could not find source path '{source}'. Please check source path")
        
    if destination is not None:
        if os.path.isabs(destination):
                destination = os.path.join(os.getcwd(),destination[1:])
        if not os.path.exists(destination):
            print(f"Destination not found. Your files will be stored in {destination}")
            os.makedirs(destination)

def ini_parser(file_name):
    ''' parser for configuration.ini file'''
    
    config = configparser.ConfigParser()
    config.read(file_name)
    sections = config.sections()
    configuration = {}
    for section in sections:
        configuration.update(dict(config[section]))
    for k,v in configuration.items():
        if v.isdigit():
            configuration[k] = int(v)
        else:
            try:
                configuration[k] = float(v)
            except ValueError:
                pass
    return configuration

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def prepare_data(config):
    '''Prepare data for the customn dataloader'''
    check_paths(source = config['dataset'])
    ratings_df = pd.read_csv(os.path.join(config['dataset'], 'ratings.csv'))
    
    # get unique users
    unique_users = ratings_df['userId'].unique()
    user_to_index = {old: new for new, old in enumerate(unique_users)} # map unique userId in the dataset to index
    new_users = ratings_df['userId'].map(user_to_index) 

    # get unique movies
    unique_movies = ratings_df.movieId.unique()
    movie_to_index = {old: new for new, old in enumerate(unique_movies)} # map unique movies in the dataset to index
    new_movies = ratings_df.movieId.map(movie_to_index)

    # make an embedding for users and movie
    n_users = unique_users.shape[0]
    n_movies = unique_movies.shape[0]
    x = np.asarray(pd.DataFrame({'user_id': new_users, 'movie_id': new_movies}))
    y = np.asarray(ratings_df["rating"])
    return (x,y),(n_users,n_movies),(user_to_index, movie_to_index),(unique_users, unique_movies)

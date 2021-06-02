import numpy as np
import pandas as pd
import os 
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.load_dataframe()
        self.prepare_data()

    def load_dataframe(self):
        self.ratings_df = pd.read_csv(os.path.join(self.data_root, 'ratings.csv'))
        self.movies_df = pd.read_csv(os.path.join(self.data_root, 'movies.csv'))
    
    def get_dataste_data(self):
        return (self.n_users, self.n_movies), (self.user_to_index, self.movie_to_index)

    def prepare_data(self):
        # get unique users
        unique_users = self.ratings_df['userId'].unique()
        self.user_to_index = {old: new for new, old in enumerate(unique_users)}
        new_users = self.ratings_df['userId'].map(self.user_to_index)

        # get unique movies
        unique_movies = self.ratings_df.movieId.unique()
        self.movie_to_index = {old: new for new, old in enumerate(unique_movies)}
        new_movies = self.ratings_df.movieId.map(self.movie_to_index)

        # make an embedding for users and movie
        self.n_users = unique_users.shape[0]
        self.n_movies = unique_movies.shape[0]
        self.x = np.asarray(pd.DataFrame({'user_id': new_users, 'movie_id': new_movies}))
        self.y = np.asarray(self.ratings_df['rating'])

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        label = torch.tensor(float(label))
        return data, label





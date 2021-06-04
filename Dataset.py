import numpy as np
import pandas as pd
import os 
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    '''Cutom data loader'''

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.load_dataframe()
        self.prepare_data()

    def load_dataframe(self):
        self.ratings_df = pd.read_csv(os.path.join(self.dataset_path, 'ratings.csv'))
        print('dataframe loaded')
    
    def get_dataset_encoding(self):
        return (self.n_users, self.n_movies), (self.user_to_index, self.movie_to_index)

    def prepare_data(self):
        # get unique users
        unique_users = self.ratings_df['userId'].unique()
        self.user_to_index = {old: new for new, old in enumerate(unique_users)} # map unique userId in the dataset to index
        new_users = self.ratings_df['userId'].map(self.user_to_index) 

        # get unique movies
        unique_movies = self.ratings_df.movieId.unique()
        self.movie_to_index = {old: new for new, old in enumerate(unique_movies)} # map unique movies in the dataset to index
        new_movies = self.ratings_df.movieId.map(self.movie_to_index)

        unique_ratings = self.ratings_df.rating.unique()
        self.ratings_to_index = {old: new for new, old in enumerate(unique_ratings)} # map unique movies in the dataset to index
        new_ratings = self.ratings_df.rating.map(self.ratings_to_index)
        

        # make an embedding for users and movie
        self.n_users = unique_users.shape[0]
        self.n_movies = unique_movies.shape[0]
        self.x = np.asarray(pd.DataFrame({'user_id': new_users, 'movie_id': new_movies}))
        self.y = np.asarray(new_ratings)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]
        label = torch.tensor(float(label)).long()
        return data, label





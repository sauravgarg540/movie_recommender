import os
import numpy as np
import pandas as pd
import torch
from utils import  ini_parser, prepare_data, configuration, check_paths

from model import EmbeddingNet



def main():
    
    configpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configuration.ini')
    config = ini_parser(configpath)

    parser = configuration()
    args = parser.parse_args()
    userId = args.userId

    # load dataframes
    ratings_df = pd.read_csv(os.path.join(config['dataset'], 'ratings.csv'))
    ratings_df = ratings_df.drop(columns = 'timestamp')
    movie_df = pd.read_csv(os.path.join(config['dataset'], 'movies.csv'))
    df = ratings_df.merge(movie_df, how = 'outer', on = 'movieId')

    (input,target),(n_users,n_movies),(user_to_index, movie_to_index),(unique_users, unique_movies) = prepare_data(config)

    # load trained weights
    check_paths(source = 'checkpoints/checkpoint.pt')
    checkpnt = torch.load("checkpoints/checkpoint.pt")
    net = EmbeddingNet(n_users, n_movies, n_factors=config['n_factors'])
    net.load_state_dict(checkpnt['model_state_dict'])
    net.cuda()

    index_to_movies = dict((v,k) for k,v in movie_to_index.items())
    index_to_users = dict((v,k) for k,v in user_to_index.items())
    encoded_user_id = user_to_index[userId]
    indices = np.where(input[:,0] == encoded_user_id)[0]

    # movies watched by the users
    x = input[input[:,0] == encoded_user_id]
    y = target[indices[0]:indices[-1]+1]
    # dataframe with real ids
    movied_liked = df[df['userId'] == userId]
    movied_liked = movied_liked.sort_values(by = 'rating', ascending=False)
    print('Top 20 movies liked by the user')
    print(movied_liked[:20])
    print('--------------------------------------------------------------------------------------------------')

    # Predict ratings fo the movies already watched by the user
    # net.eval()
    # with torch.no_grad():
    #     x = torch.from_numpy(x).cuda()
    #     y = torch.from_numpy(y).cuda()
    #     output = net.forward(x[:,0],x[:,1])
    # output = output.detach().cpu().numpy()
    # x = x.detach().cpu().numpy()

    # predicted_movies = pd.DataFrame()
    # predicted_movies['userId']= x[:,0]
    # predicted_movies['userId']= predicted_movies['userId'].map(index_to_users)
    # predicted_movies['movieId']= x[:,1]
    # predicted_movies['movieId']= predicted_movies['movieId'].map(index_to_movies)
    # predicted_movies['rating'] = output
    # predicted_movies = predicted_movies.merge(movie_df, on = 'movieId',how = 'left')
    # predicted_movies = predicted_movies.sort_values(by = 'rating', ascending=False)
    # print('Top 20 movies liked by the user predicted by the model')
    # print(predicted_movies[:20])
    # print('--------------------------------------------------------------------------------------------------')


    # let's see movies that user may like
    movies_not_watched = np.array([i for i in unique_movies if i not in movied_liked['movieId'].tolist()])
    predicted_movies = pd.DataFrame()
    predicted_movies['userId'] = [userId]*movies_not_watched.shape[0]
    predicted_movies['movieId']= movies_not_watched
    predicted_movies['movieId']= predicted_movies['movieId'].map(movie_to_index)
    x = np.asarray(pd.DataFrame({'user_id': predicted_movies['userId'], 'movie_id': predicted_movies['movieId']}))

    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(x).cuda()
        output = net.forward(x[:,0],x[:,1])

    output = output.detach().cpu().numpy()
    predicted_movies['rating'] = output
    predicted_movies['movieId'] = predicted_movies['movieId'].map(index_to_movies)
    predicted_movies = predicted_movies.merge(movie_df, on = 'movieId',how = 'left')
    predicted_movies = predicted_movies.sort_values(by = 'rating', ascending=False)
    print('Top 20 movies user may like as predicted by the model')
    print(predicted_movies[:20])



if __name__ == '__main__':
    main()



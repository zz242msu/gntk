import numpy as np
import scipy
from multiprocessing import Pool
from os.path import join
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from util import load_data
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def search(dataset, data_dir):
    # Load the Gram matrix and labels
    gram = np.load(join(data_dir, 'gram.npy'))
    labels = np.load(join(data_dir, 'labels.npy'))
    
    # Load the indices for the train and test folds
    train_fold_idx = [np.loadtxt('dataset/{}/10fold_idx/train_idx-{}.txt'.format(dataset, i)).astype(int) for i in range(1, 11)]
    test_fold_idx = [np.loadtxt('dataset/{}/10fold_idx/test_idx-{}.txt'.format(dataset, i)).astype(int) for i in range(1, 11)]

    # Define the range of C values for the GridSearch
    C_list = np.logspace(-2, 4, 120)

    # Initialize the SVM with precomputed kernel
    svc = SVC(kernel='precomputed', cache_size=16000, max_iter=500000)

    # Apply a StandardScaler or MinMaxScaler to the data
    # scaler = StandardScaler()  # or 
    scaler = MinMaxScaler()
    gram = scaler.fit_transform(gram)
    
    # Perform Grid Search with the original/optionally scaled Gram matrix
    clf = GridSearchCV(svc, {'C': C_list}, cv=zip(train_fold_idx, test_fold_idx), n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(gram, labels)

    # Collect results
    df = pd.DataFrame({'C': C_list, 'train': clf.cv_results_['mean_train_score'], 'test': clf.cv_results_['mean_test_score']}, columns=['C', 'train', 'test'])
    df['normalized'] = False

    # Normalizing the Gram matrix in the original way
    gram_nor = np.copy(gram)
    gram_diag = np.sqrt(np.diag(gram_nor))
    gram_nor /= gram_diag[:, None]
    gram_nor /= gram_diag[None, :]

    clf = GridSearchCV(svc, {'C': C_list}, cv=zip(train_fold_idx, test_fold_idx), n_jobs=80, verbose=0, return_train_score=True)

    # Perform Grid Search with the normalized Gram matrix
    clf.fit(gram_nor, labels)
    df_nor = pd.DataFrame({'C': C_list, 'train': clf.cv_results_['mean_train_score'], 'test': clf.cv_results_['mean_test_score']}, columns=['C', 'train', 'test'])
    df_nor['normalized'] = True

    # Concatenate results and save to CSV
    all_df = pd.concat([df, df_nor])[['C', 'normalized', 'train', 'test']]
    all_df.to_csv(join(data_dir, 'grid_search.csv'))
       
parser = argparse.ArgumentParser(description='hyper-parameter search')
parser.add_argument('--data_dir', type=str, required=True, help='data_dir')
parser.add_argument('--dataset', type=str, required=True, help='dataset')
args = parser.parse_args()
search(args.dataset, args.data_dir)

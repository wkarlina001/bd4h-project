import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import NearestNeighbors

def clean_data(df):
    """Clean input Data - Remove Extra Characters; Encode Categorical Values; Prepare and Filter Target Values

    Parameters
    ----------
    df : input dataframe
    """
    #check for missing values
    # print(df.isnull().sum()) # to verify

    # remove extra letters in disease diagnosis
    df['Insominia'] = df['Insominia'].astype(str).str[0]
    df['shizopherania'] = df['shizopherania'].astype(str).str[0]
    df['vascula_demetia'] = df['vascula_demetia'].astype(str).str[0]
    df['ADHD'] = df['ADHD'].astype(str).str[0]
    df['Bipolar'] = df['Bipolar'].astype(str).str[0]

    #apply one hot encoding to categorical features
    enc = OrdinalEncoder()
    df[["sex", "faNoily_status", "religion" ,"occupation" , "genetic", "status", "loss_of_parent", "divorse", "Injury", "Spiritual_consult"]] = enc.fit_transform(df[["sex", "faNoily_status", "religion" ,"occupation" , "genetic", "status", "loss_of_parent", "divorse", "Injury", "Spiritual_consult"]])

    # prepare output label into multiclass classification
    label_dict = {'N': 0 , 'P': 1}
    category = ['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar']
    for cat in category:
        df[cat+"_enc"] = df[cat].map(label_dict)
    # df.drop(['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar'], axis=1, inplace=True)

    # create combined label
    df["target"] = df['Insominia_enc'].astype(str) + df['shizopherania_enc'].astype(str) + df['vascula_demetia_enc'].astype(str) + df['ADHD_enc'].astype(str) + df['Bipolar_enc'].astype(str)

    # count of multilabel
    df_target_count = df['target'].value_counts()
    df_filter = df[~df['target'].isin(df_target_count[df_target_count < 6].index)]
    df_target_count = df_filter['target'].value_counts().to_frame().reset_index() # to verify
    # print(df_target_count)

    return df, df_filter, df_target_count

def prepare_data(df_target_count, df_filter, SEED, smote=False):
    """Prepare Features and Target with/without SMOTE 

    Parameters
    ----------
    df_target_count : dataframe contains count of target label
    df_filter : dataframe after removing target with less than 7 count
    smote : boolen - option to turn on/off SMOTE 
    """
    # prepare data for training and testing
    target_dict = {}
    target_list = df_target_count['index'].values
    for i in range(len(target_list)):
        target_dict[target_list[i]] = i
    # print(target_dict)

    if smote is True:
        # prepare dataset with SMOTE applied
        max_target_sample = df_target_count['target'].max()
        target_list = df_target_count['index'].tolist() #target list for oversample to max_target_sample
        Y_smote = np.empty((0))
        X_smote =  np.empty((0, 11)) #hardcode no of features

        for target in target_list:
            df1 = df_filter.loc[df_filter['target'] == target]

            # split features and target
            X = df1.drop(['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar', 'Insominia_enc', 'shizopherania_enc', 'vascula_demetia_enc', 'ADHD_enc', 'Bipolar_enc', 'target', 'agecode'], axis=1)
            feature_category = X.columns.tolist()

            X_array = X.to_numpy()
            Y = [target] * max_target_sample

            sm = SMOTE(X_array, k_neigh = 5, random_state = SEED).fit()
            samples = sm.sample(max_target_sample)
            # print(samples.shape, len(Y), target)

            # append to final X features and Y target
            X_smote = np.append(X_smote, samples, 0)
            Y_smote = np.append(Y_smote, Y, 0)

        # Y_smote = np.vectorize(target_dict.get)(Y_smote)
        Y_arr = []
        for y_val in Y_smote:
            Y_arr.append(np.array(list(map(int, y_val))))
        Y_smote = (np.array(Y_arr))
        return X_smote, Y_smote, feature_category
        
    else:
        # prepare dataset without SMOTE applied
        X = df_filter.drop(['Insominia', 'shizopherania', 'vascula_demetia', 'ADHD', 'Bipolar', 'Insominia_enc', 'shizopherania_enc', 'vascula_demetia_enc', 'ADHD_enc', 'Bipolar_enc', 'target', 'agecode'], axis=1)
        feature_category = X.columns.tolist()
        
        X = X.to_numpy()
        Y = df_filter['target'].values
        # Y = np.vectorize(target_dict.get)(Y)
        Y_arr = []
        for y_val in Y:
            Y_arr.append(np.array(list(map(int, y_val))))
        Y = (np.array(Y_arr))
        return X, Y, feature_category

class SMOTE:
    # apply SMOTE
    # https://towardsdatascience.com/imbalanced-classification-in-python-smote-enn-method-db5db06b8d50
    # https://github.com/dialnd/imbalanced-algorithms/blob/master/smote.py
    # https://scikit-learn.org/stable/modules/neighbors.html
    """Implementation of Modified Synthetic Minority Over-Sampling Technique (SMOTE)

    Parameters
    ----------
    X : array-like , shape = [n_samples, n_features]
        Original data before oversampling
    k_neighbors : int, optional (default=5)
        Number of nearest neighbors.
    random_state : int or optional (default=123)
    
    """
    def __init__(self, X, k_neigh = 5, random_state = 123):
        self.X = X

        self.k_neigh = k_neigh
        self.random_state = random_state

    def fit(self):
        """Train KNN model with input data"""
        
        # Learn nearest neighbors.
        self.neigh = NearestNeighbors(n_neighbors=self.k_neigh).fit(self.X)
        return self
    
    def sample(self, n_sample):
        """Generate samples to n_sample amount.

        Parameters
        ----------
        n_samples : int
            Number of new synthetic samples.
        Returns
        -------
        sample : array, shape = [n_samples, n_features]
            Returns synthetic samples.
        """
        np.random.seed(seed=self.random_state)
        if (self.X.shape[0] >= n_sample):
            S = self.X[:n_sample, :]
            
            return S

        #initialise array for samples
        S = np.zeros(shape=(n_sample, self.X.shape[1]))
        for i in range(n_sample):
            rand = np.random.random(1)[0]
            j = np.random.randint(0, self.X.shape[0]) #random index of minority sample

            # Find the NN for each sample. Remove sample from indice
            _ , indice = self.neigh.kneighbors(self.X[j].reshape(1, -1))
            indice = indice[:, 1:]

            knn_index = np.random.choice(indice[0])
            # print(j, indice, knn_index, self.X[j].reshape(1, -1))
            diff = self.X[knn_index] - self.X[j]
            S[i, :] = self.X[j, :] + rand * diff[:]

        return S
        
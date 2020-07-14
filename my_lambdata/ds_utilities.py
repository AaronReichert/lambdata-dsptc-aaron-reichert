import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from pdb import set_trace as breakpoint
from IPython.display import display


class My_Data_Splitter():
    def __init__(self, df, features, target):
        self.df = df
        self.features = features
        self.target = target
        self.X = df[features]
        self.y = df[target]

    def train_validation_test_split(self,
                                    train_size=0.7, val_size=0.1,
                                    test_size=0.2, random_state=None,
                                    shuffle=True):
        """
        This function is a utility wrapper around the Scikit-Learn train_test_split that splits arrays or 
        matrices into train, validation, and test subsets.
        Args:
            X (Numpy array or DataFrame): This is a dataframe with features.
            y (Numpy array or DataFrame): This is a pandas Series with target.
            train_size (float or int): Proportion of the dataset to include in the train split (0 to 1).
            val_size (float or int): Proportion of the dataset to include in the validation split (0 to 1).
            test_size (float or int): Proportion of the dataset to include in the test split (0 to 1).
            random_state (int): Controls the shuffling applied to the data before applying the split for reproducibility.
            shuffle (bool): Whether or not to shuffle the data before splitting
        Returns:
            Train, test, and validation dataframes for features (X) and target (y). 
        """
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size / (train_size + val_size),
            random_state=random_state, shuffle=shuffle)
        return X_train, X_val, X_test, y_train, y_val, y_test
    def print_split_summary(self, X_train, X_val, X_test):
        print('######################## TRAINING DATA ########################')
        print(f'X_train Shape: {X_train.shape}')
        display(X_train.describe(include='all').transpose())
        print('')
        print('######################## VALIDATION DATA ######################')
        print(f'X_val Shape: {X_val.shape}')
        display(X_val.describe(include='all').transpose())
        print('')
        print('######################## TEST DATA ############################')
        print(f'X_test Shape: {X_test.shape}')
        display(X_test.describe(include='all').transpose())
        print('')

def tvt_split(df, tvt_stratify='target', tvt_train_size=0.70,
              tvt_val_size=0.15, tvt_test_size=0.15, tvt_random_state=42):
    '''This function uses train test split and calculates an extra split for
    your validation set.
    It also stratifies and splits everything into X and y sets.
    example:
    _train, y_train, X_val, y_val, X_test, y_test=tvt_split(
    df,
    tvt_stratify='target_column'
    )
    '''
    tvt_df = df.copy()
    tvt_temp_size = tvt_val_size + tvt_test_size
    train, temp = train_test_split(tvt_df, train_size=tvt_train_size,
                                   test_size=tvt_temp_size,
                                   stratify=tvt_df[tvt_stratify],
                                   random_state=tvt_random_state)
    tvt_val_size_adjusted = tvt_val_size / tvt_temp_size
    tvt_test_size_adjusted = tvt_test_size / tvt_temp_size
    val, test = train_test_split(temp, train_size=tvt_val_size_adjusted,
                                 test_size=tvt_test_size_adjusted,
                                 stratify=temp[tvt_stratify],
                                 random_state=tvt_random_state)
    X_train = train.drop(tvt_stratify, axis=1)
    y_train = train[tvt_stratify]
    X_val = val.drop(tvt_stratify, axis=1)
    y_val = val[tvt_stratify]
    X_test = test.drop(tvt_stratify, axis=1)
    y_test = test[tvt_stratify]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_validation_test_split(X, y, train_size=0.7, val_size=0.1,
                                test_size=0.2, random_state=None,
                                shuffle=True):

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/(train_size+val_size),
        random_state=random_state, shuffle=shuffle)

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':

    raw_data = load_wine()
    df = pd.DataFrame(data=raw_data['data'], columns=raw_data['feature_names'])
    df['target'] = raw_data['target']
    # breakpoint()

    # X_train, X_val, X_test, y_train, y_val, y_test = train_validation_test_split(
    #     df[['ash', 'hue']], df['target'])
    #
    # Test the My_Data_Splitter class
    splitter = My_Data_Splitter(df=df, features=['ash', 'hue'], target='target')
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.train_validation_test_split
    splitter.print_split_summary(X_train, X_val, X_test)

import pandas as pd
from sklearn.model_selection import train_test_split

def tvt_split(df, tvt_stratify='target', tvt_train_size=0.70, tvt_val_size=0.15, tvt_test_size=0.15, tvt_random_state=42):
  ''' first copy
  split off the train
  split the rest between val and test
  split the X and y'''
  tvt_df=df.copy() 
  tvt_temp_size=tvt_val_size+tvt_test_size
  train, temp = train_test_split(tvt_df, train_size=tvt_train_size, test_size=tvt_temp_size, 
                            stratify=tvt_df[tvt_stratify], random_state=tvt_random_state)
  tvt_val_size_adjusted=tvt_val_size/tvt_temp_size
  tvt_test_size_adjusted=tvt_test_size/tvt_temp_size
  val, test = train_test_split(temp, train_size=tvt_val_size_adjusted, test_size=tvt_test_size_adjusted, 
                            stratify=temp[tvt_stratify], random_state=tvt_random_state)  
  X_train=train.drop(tvt_stratify,axis=1)
  y_train=train[tvt_stratify]
  X_val=val.drop(tvt_stratify,axis=1)
  y_val=val[tvt_stratify]
  X_test=test.drop(tvt_stratify,axis=1)
  y_test=test[tvt_stratify]

  return X_train, y_train, X_val, y_val, X_test, y_test
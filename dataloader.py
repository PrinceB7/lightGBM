from lightgbm import Dataset
import pandas as pd
import pickle


def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
    
def load_df(path):
    df = pd.read_csv(path, encoding='utf-8')
    return df

def load_feather(path):
    df = pd.read_feather(path)
    return df
    
def split_label(df, label_pos=-1, key_pos=0):
    k = df.iloc[:, key_pos]
    y = df.iloc[:, label_pos]
    x = df.iloc[:, key_pos+1:label_pos]
    
    return k, x, y

def convert_lgb_ds(x_df, y_df):
    return Dataset(x_df, label=y_df, free_raw_data=False)

# def make_lgb_ds(path, feather=True):
#     if feather == True:
#         df = load_feather(path)
#     else:
#         df = load_df(path)
#     x, y = split_label(df, label_pos=-1)
#     ds = convert_lgb_ds(x, y)
#     return ds



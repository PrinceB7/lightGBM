
from sklearn.preprocessing import LabelEncoder
import numpy as np

def select_column(df):
    col_list = list(df.columns)

    key_col_list = ['TRD_NO']  # 키
    target_col_list = ['target']  # 타겟
     
    # 학습시 제외할 컬럼
    except_col_list = ['CP_CD', 'CP_NM', 'GODS_NM', 'PAYR_SEQ',
                       'MPHN_NO', 'PAYR_IP', 'SUB_IP_A',  'SUB_IP_B',
                       'SUB_IP_C', 'SUB_IP_D', 'FGPT']
    # 날짜 관련 컬럼
    date_col_list= ['REQ_DD', 'PAY_YM']

    select_col_list = [col for col in col_list if col not in except_col_list]
    select_col_list = [col for col in select_col_list if col not in date_col_list]
    
    return df[select_col_list]

def get_imputation_dict(df):
    imputation_dict = dict()
    
    # 결측치(object)
    null_col_object_list = ['NPAY_YN', 'PAY_MTHD_CD', 'ARS_AUTHTI_YN', 'CP_M_CLF_NM', 'CP_S_CLF_NM']
    # 결측치(float)
    null_col_num_list = ['MM_LMT_AMT', 'REMD_LMT_AMT']
    
    for col in null_col_object_list:

        mode = df[col].mode().values[0]
        imputation_dict[col] = mode

    for col in null_col_num_list:
        imputation_dict[col] = 0
        
    return imputation_dict


def impute(df, imputation_dict):
    
    for key, val in imputation_dict.items():
        df[key].fillna(val, inplace=True)
        
    return df
        
def get_encoder_dict(df):
    # object type 컬럼
    object_col_list = ['COMMC_CLF', 'NPAY_YN', 'PAY_MTHD_CD',
                       'ARS_AUTHTI_YN', 'GNDR', 'FOREI_YN',
                       'AUTHTI_CLF_FLG', 'SVC_CLF_NM', 'CP_M_CLF_NM', 'CP_S_CLF_NM']
    
    encoder_dict = {col: LabelEncoder() for col in object_col_list}
    
    for col in object_col_list:
        # Fit encoder
        encoder_dict[col].fit(df[col])
        
    return encoder_dict

def encode(df, encoder_dict):
    
    for col, encoder in encoder_dict.items():
        for label in np.unique(df[col]):
            if label not in encoder.classes_:
                encoder.classes_ = np.append(encoder.classes_, label)
        df[col] = encoder.transform(df[col])  
    return df

def preprocess(train_df, test_df):
    
    train_df, test_df = select_column(train_df), select_column(test_df)
    imputation_dict = get_imputation_dict(train_df)
    
    train_df, test_df = impute(train_df, imputation_dict), impute(test_df, imputation_dict)
    
    encoder_dict = get_encoder_dict(train_df)
    
    train_df =  encode(train_df, encoder_dict)
    test_df =  encode(test_df, encoder_dict)
    
    return train_df, test_df, imputation_dict, encoder_dict

def train_val_split(df, val_ym='201910'):
    val_idx = df['REQ_DD'].apply(lambda x: str(x)[:6]) == val_ym
    val_df = df[val_idx]
    train_df = df[~val_idx]
    print(f"Train: {train_df.shape}, Validation: {val_df.shape}")
    return train_df, val_df

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
import xgboost as xgb
from argparse import ArgumentParser
from dataloader import convert_lgb_ds, load_feather, load_pickle, save_pickle, split_label
from preprocessing import *
from model import LightGBM
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':

    args = ArgumentParser()
    args.add_argument("--mode", type=str, default='train')
    args.add_argument("--data_path", type=str, default='../../data/.train/.task150/')
    args.add_argument("--num_iterations", type=int, default=3000)
    args.add_argument("--early_stopping_rounds", type=int, default=100)
    args.add_argument("--model_name", type=str, default='model.pkl')
    args.add_argument("--prediction_file", type=str, default='submission.feather')

    config = args.parse_args()
    
    mode = config.mode
    DATA_PATH = config.data_path
    num_iterations = config.num_iterations
    early_stopping_rounds = config.early_stopping_rounds
    model_name = 'model/' + config.model_name
    prediction_file = 'prediction/' + config.prediction_file
    
    print("\nI loaded everything...\n")
    
    if mode == 'train':
    
        train_df = load_feather(DATA_PATH+'train.feather')
        train_df, val_df = train_val_split(train_df, val_ym='201907')
                
        train_df, val_df, imputation_dict, encoder_dict = preprocess(train_df, val_df)
            
        key_train, x_train, y_train = split_label(train_df)
        key_val, x_val, y_val = split_label(val_df)
        
        train_ds = convert_lgb_ds(x_train, y_train)
        validate_ds = convert_lgb_ds(x_val, y_val)
        
        #'''
        # Train
        print("\nStarting training...\n")
        model = LightGBM(model_name=model_name)
        model.load_model('model/model_bese.pkl')
        
        model.train(train_ds=train_ds,
                    val_ds=validate_ds, 
                    num_boost_round=num_iterations, 
                    early_stopping_rounds=early_stopping_rounds, 
                    verbose_eval=100)
        
        print("\nFinished training!!!\n")
        
        save_pickle('encoder/imp_dict.pkl', imputation_dict)
        save_pickle('encoder/enc_dict.pkl', encoder_dict)
        #'''
        
        '''
        model = LightGBM(model_name=model_name)
        model.load_model('model/model_98_46.pkl')
        '''
        
        
        
        # Result
        #val_pred = model.predict(validate_ds.data)
        #save_result(val_pred, path=prediction_file)
        #val_pred = load_result(path=prediction_file)
        
        #print("Validation auc: {}".format(roc_auc_score(validate_ds.label, val_pred)))
        
        #train_pred = model.predict(train_ds.data)
        
        #print("Train auc: {}".format(roc_auc_score(train_ds.label, train_pred)))

        
    elif mode == 'test':
        
        model = LightGBM()
        model.load_model(model_name)
        #model = model.model
        
        x_test = load_feather(DATA_PATH+'test.feather')
        imputation_dict = load_pickle('encoder/imp_dict.pkl')
        encoder_dict = load_pickle('encoder/enc_dict.pkl')
        
        x_test = select_column(x_test)
        x_test = impute(x_test, imputation_dict)
        x_test = encode(x_test, encoder_dict)
        
        key_test = x_test.iloc[:, 0]
        x_test = x_test.iloc[:, 1:]
              
        test_pred = model.predict(x_test)
        #print(test_pred[:20], test_pred.shape)
        
        sub_df = pd.read_feather('../../data/.sample_submission/.task150/sample_submission.feather')
        sub_df['target'] = test_pred
        sub_df.to_feather(prediction_file)
        
        print("\nDone!\n")
        
        # save_result(test_pred, path=prediction_file)

    
    elif mode == 'trainx':
    
        train_df = load_feather(DATA_PATH+'train.feather')
        train_df, val_df = train_val_split(train_df, val_ym='2019070')
                
        train_df, val_df, imputation_dict, encoder_dict = preprocess(train_df, val_df)
            
        key_train, x_train, y_train = split_label(train_df)
        key_val, x_val, y_val = split_label(val_df)
        
        #train_ds = convert_lgb_ds(x_train, y_train)
        #validate_ds = convert_lgb_ds(x_val, y_val)
        
        #'''
        # Train
        print("\nStarting training xgboost...\n")
        
        eval_set = [(x_val, y_val)]
        model = xgb.XGBClassifier(max_depth=50, min_child_weight=1, n_estimators=100, objective='binary:logistic', verbose=1, learning_rate=0.3)
        #model = xgb.train(params, dtrain, num_boost_round=1000, early_stopping_rounds=100, verbose_eval=50, feval='auc')
        model.fit(x_train, y_train, eval_set=eval_set, eval_metric='auc', early_stopping_rounds=10)
        
        model.save_model('model/xgb1.pkl')
        
        print("\nFinished training!!!\n")
        
        save_pickle('encoder/imp_dict.pkl', imputation_dict)
        save_pickle('encoder/enc_dict.pkl', encoder_dict)
        #'''
        
        '''
        model = LightGBM(model_name=model_name)
        model.load_model('model/model_98_46.pkl')
        '''
        
        
        
        # Result
        val_pred = model.predict(x_val)
        #save_result(val_pred, path=prediction_file)
        #val_pred = load_result(path=prediction_file)
        
        print("Validation auc: {}".format(roc_auc_score(y_val, val_pred)))
        
        train_pred = model.predict(x_train)
        
        print("Train auc: {}".format(roc_auc_score(y_train, train_pred)))
        
        
    elif mode == 'testx':
        
        model = xgb.XGBClassifier()
        model.load_model('model/xgb1.pkl')
        #model = model.model
        
        x_test = load_feather(DATA_PATH+'test.feather')
        imputation_dict = load_pickle('encoder/imp_dict.pkl')
        encoder_dict = load_pickle('encoder/enc_dict.pkl')
        
        x_test = select_column(x_test)
        x_test = impute(x_test, imputation_dict)
        x_test = encode(x_test, encoder_dict)
        
        key_test = x_test.iloc[:, 0]
        x_test = x_test.iloc[:, 1:]
              
        test_pred = model.predict(x_test)
        #print(test_pred[:20], test_pred.shape)
        
        sub_df = pd.read_feather('../../data/.sample_submission/.task150/sample_submission.feather')
        sub_df['target'] = test_pred
        sub_df.to_feather(prediction_file)
        
        print("\nDone!\n")
        
        # save_result(test_pred, path=prediction_file)    
    
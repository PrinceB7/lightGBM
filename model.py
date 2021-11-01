"""
model.py
"""
import lightgbm as lgb
import pickle

class LightGBM:
    
    def __init__(self, model_name='model.pkl'):
        self.params = {'learning_rate': 0.01,
          'num_leaves': 144, 
          'boosting': 'gbdt', 
          'objective': 'binary', 
          'metric': 'auc', 
          'feature_fraction': 0.9, 
          'bagging_fraction': 0.9, 
          'bagging_freq': 5, 
          'seed':42}
        self.model_name = model_name
        
    def train(self, train_ds, val_ds,
              num_boost_round=1000, early_stopping_rounds=100, verbose_eval=100):
        self.model = lgb.train(self.params,
                               train_set=train_ds,
                               valid_sets=[val_ds],
                               num_boost_round=num_boost_round,
                               verbose_eval=verbose_eval,
                               early_stopping_rounds=early_stopping_rounds)
        #self.save_model()
        
    def predict(self, x):
        y_pred = self.model.predict(x)
        return y_pred    
    
    def load_model(self, path):
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            print("Load model: {}".format(path))
            
        except Exception as e:
            print("Fail to Load model {}".format(e))

        
    def save_model(self):
        path = self.model_name
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)
            print("Saved model: {}".format(path))
            
        except Exception as e:
            print("Fail to save model {}".format(e))

        
        
    
        
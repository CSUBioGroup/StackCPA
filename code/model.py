import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import Imputer
from sklearn.model_selection import KFold
from math import sqrt
from scipy import stats
import numpy as np
import catboost as cat
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn import tree
# import pickle
from joblib import dump, load

def regression_scores(label, pred):
    label = np.array(label).reshape(-1)
    pred = np.array(pred).reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    pearson = np.corrcoef(label, pred)[0, 1]
    spearman = stats.spearmanr(label, pred)[0]
    
    #cal cindex
    idx = np.argsort(label)
    label,pred = label[idx],pred[idx]
    pair,sum = 0.0,0.0
    for i in range(len(label)-1,0,-1):
        for j in range(i-1,-1,-1):
            if label[i] <= label[j]:
                continue
            pair += 1
            if pred[i] < pred[j]:
                continue
            elif pred[i] == pred[j]:
                sum += 0.5
                continue    
            sum +=1
    ci = sum/pair
    
    return round(rmse, 6), round(pearson, 6), round(spearman, 6), round(ci,6)


class CpaStacking:
    def __init__(self, model_set=None,final_model=None,n_folds=5):
        self.model_set = model_set
        self.final_model = final_model
        self.trained_model_set = None
        self.trained_final_model = None
        self.n_folds = n_folds                  # the number of fold for cross validation


    def trained_by_muti_model(self,train_data,train_label,layer_num):
        trained_model_set = []
        # best_rmse = [0xfffff for i in range(len(self.model_set[0]))]
        retrain_label = np.zeros((train_data.shape[0]))
        #2nd layer concate 6 dim vector for the 3nd layer
        if layer_num == 1:
            retrain_data = np.zeros((train_data.shape[0],len(self.model_set[0])))
        else:
            retrain_data = np.zeros((train_data.shape[0],2*len(self.model_set[0])))

        sub_retrain_data = np.zeros((train_data.shape[0],len(self.model_set[0])))
        for i,model in enumerate(self.model_set[layer_num-1]):
            model.fit(train_data,train_label)
            pred_res = model.predict(train_data)
            
            trained_model_set.append(model)
            sub_retrain_data[:,i] = pred_res 
    
        print(sub_retrain_data.shape,sub_retrain_data[0])

        if layer_num == 1:
            retrain_data = sub_retrain_data
        else:
            retrain_data = np.hstack((sub_retrain_data,train_data))
        retrain_label = train_label
            
        return retrain_data,retrain_label,trained_model_set

    def trained_by_regression_model(self,train_data,train_label):
        self.final_model.fit(train_data,train_label)



    #for cold start 
    def train(self,train_data,train_label,test_data,test_label):
        model_set = []

        layer1_data, layer1_label,layer1_model = self.trained_by_muti_model(train_data,train_label,1)
        print('first layer training is finished')
        model_set.append(layer1_model)

        # 2nd stacking
        layer2_data, layer2_label,layer2_model = self.trained_by_muti_model(layer1_data,layer1_label,2)
        print('second layer training is finished')
        model_set.append(layer2_model)

        # 3nd predictor 
        print('third layer training is finished')
        self.trained_by_regression_model(layer2_data, layer2_label)
            
        pred_res = np.zeros((test_data.shape[0],0))
        for idx in range(2):
            sub_data = np.zeros((test_data.shape[0],len(self.model_set[0])))
            for i,model in enumerate(model_set[idx]):
                if idx == 0:
                    sub_data[:,i] = model.predict(test_data)
                else:
                    sub_data[:,i] = model.predict(pred_res)

            pred_res = np.hstack((pred_res,sub_data))
        
        pred_final_res = self.final_model.predict(pred_res)

        rmse, pearson, spearman,ci = regression_scores(test_label,pred_final_res)
        
        self.trained_model_set = [layer1_model,layer2_model]
        self.trained_final_model = self.final_model
    
        return self,[rmse, pearson, spearman, ci]

    def predict_by_muti_model(self,data,layer_num):
        prediction = np.zeros((data.shape[0],len(self.trained_model_set[layer_num-1])))
                              
        for idx,model in enumerate(self.trained_model_set[layer_num-1]):
            pred = model.predict(data)
            prediction[:,idx] = pred
        return prediction
        
    def predict(self,data):
        print('start predict')
        #layer1
        print('first layer prediction is finished')
        l1_prediction = self.predict_by_muti_model(data,1) 
        #layer2
        l2_prediction = self.predict_by_muti_model(l1_prediction,2)
        print('first layer prediction is finished')
        #layer3
        l3_data = np.concatenate((l1_prediction,l2_prediction),axis=1)
        prediction = self.trained_final_model.predict(l3_data)
        print('prediction finishied\n')
        return prediction
    
    @staticmethod    
    def load_model(file):
        with open(file,'rb') as model_file:
            return load(model_file)
            
    def save_model(self,file_path):
        with open(file_path,'wb') as model_file:
            dump(self,model_file)

        
        

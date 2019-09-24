#!/usr/bin/env python
#-*- coding: utf-8 -*-

import pandas as pd
import numpy as np 
from getdata import getData
from input_fn import HousePriceDataset,FeatureEngineer
from sklearn.model_selection import KFold, cross_val_score
from modeling import MyModel,predict_all,rmsle


if __name__ == '__main__':
        # loading data
    df_train = getData('data/train.csv')
    df_test = getData('data/test.csv')

    # preprocessing
    h = HousePriceDataset(df_train,df_test)
    cleaned, y_train = h.mainClean()

    # engineering
    engineer = FeatureEngineer(cleaned,y_train)
    cleaned, y_train, X_test = engineer.main()


    # modeling
    kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
    models = MyModel(cleaned,y_train,kfolds)
    elastic, lasso,ridge, svr, gbr, sgb, lgb, stack = models.main()
    model_list = [elastic, lasso,ridge, svr, gbr, sgb, lgb, stack]


    # evaluation

    print('RMSLE score on train data:')
    print(rmsle(y_train,predict_all(cleaned,model_list)))

    # Give result
    final_prediction = predict_all(X_test,model_list)
    df_test['final_prediction_price'] = final_prediction
    df_test.to_csv("submission.csv", index=False)
    print('Mission Complete !!')










from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from datetime import datetime
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')



def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X,y,cv):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=cv))
    return (rmse)

class MyModel(object):
    def __init__(self,X,y,kfolds):
        super(MyModel,self).__init__()
        self.alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
        self.alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
        self.e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
        self.e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
        self.X = X
        self.y = y
        self.kfolds = kfolds

    def lgm(self):
        lightgbm = LGBMRegressor(objective='regression', 
                                            num_leaves=4,
                                            learning_rate=0.01, 
                                            n_estimators=5000,
                                            max_bin=200, 
                                            bagging_fraction=0.75,
                                            bagging_freq=5, 
                                            bagging_seed=7,
                                            feature_fraction=0.2,
                                            feature_fraction_seed=7,
                                            verbose=-1,
                                            )
        lightgbm.fit(self.X,self.y)
        return lightgbm


    def ridge(self):
        ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=self.alphas_alt, cv=self.kfolds))
        ridge.fit(self.X,self.y)
        return ridge

    def lasso(self):

        lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=self.alphas2, random_state=42, cv=self.kfolds))
        lasso.fit(self.X,self.y)
        return lasso

    def elas(self):

        elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=self.e_alphas, cv=self.kfolds, l1_ratio=self.e_l1ratio))
        elasticnet.fit(self.X,self.y)
        return elasticnet

    def svr(self):

        svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
        svr.fit(self.X,self.y)
        return svr

    def gbr(self):

        gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4,
         max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                       
        gbr.fit(self.X,self.y)
        return gbr

    def xgb(self):

        xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                            max_depth=3, min_child_weight=0,
                                            gamma=0, subsample=0.7,
                                            colsample_bytree=0.7,
                                            objective='reg:linear', nthread=-1,
                                            scale_pos_weight=1, seed=27,
                                            reg_alpha=0.00006)
        xgboost.fit(self.X.values,self.y.values)
        return xgboost


    def stack(self,ridge, lasso, elasticnet, gbr, xgboost, lightgbm):
        stackgen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),
                                        meta_regressor=xgboost,
                                        use_features_in_secondary=True)

        stackgen.fit(self.X,self.y)
        return stackgen

    def main(self):
        print('START Fit')

        print(datetime.now(), 'elasticnet')
        elastic = self.elas()
        score = cv_rmse(elastic,self.X,self.y,self.kfolds)
        print("ELASTICNET: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'lasso')
        lasso = self.lasso()
        score = cv_rmse(lasso,self.X,self.y,self.kfolds)
        print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'ridge')
        ridge = self.ridge()
        score = cv_rmse(ridge,self.X,self.y,self.kfolds)
        print("RIDGE: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'svr')
        svr = self.svr()
        score = cv_rmse(svr,self.X,self.y,self.kfolds)
        print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'GradientBoosting')
        gbr = self.gbr()
        score = cv_rmse(gbr,self.X,self.y,self.kfolds)
        print("GRADIENT: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'xgboost')
        xgb = self.xgb()
        score = cv_rmse(xgb,self.X,self.y,self.kfolds)
        print("XGBOOST: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'lightgbm')
        lgb = self.lgm()
        score = cv_rmse(lgb,self.X,self.y,self.kfolds)
        print("LIGHTBGM: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

        print(datetime.now(), 'StackingCVRegressor')
        stack = self.stack(ridge=ridge,lasso=lasso, elasticnet=elastic, gbr=gbr, xgboost=xgb, lightgbm=lgb)

        print('Our model has elasticnet, lasso, ridge, svr, gbr, xgboost, lightgbm and Stacking')
        return elastic, lasso, ridge, svr, gbr, xgb, lgb, stack


    def save_model(self, model, path):
	    with open(path, 'wb') as clf:
	        pickle.dump(model, clf) 

def predict_all(X_test,models):
    final = np.zeros((X_test.shape[0],len(models)))
    for i,v in enumerate(models):
        final[:,i] = v.predict(X_test.values)
    
    final[:,-1] = final[:,-1] * 2
    return np.sum(final,axis=1)/(len(models)+1)


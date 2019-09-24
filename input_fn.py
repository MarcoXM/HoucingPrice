import numpy as np
import pandas as pd

class HousePriceDataset(object):
    def __init__(self,train,test):
        super(HousePriceDataset,self).__init__()
        train.drop(['Id'], axis=1, inplace=True)
        test.drop(['Id'], axis=1, inplace=True)
        train = train[train.GrLivArea < 4500]
        train.reset_index(drop=True, inplace=True)
        train["SalePrice"] = np.log1p(train["SalePrice"])
        y_train = train['SalePrice'].reset_index(drop=True)
        train_features = train.drop(['SalePrice'], axis=1)
        test_features = test
        features = pd.concat([train_features, test_features]).reset_index(drop=True)

        self.features = features
        self.y_train = y_train


    def toCat(self):
        features_list = ['MSSubClass','YrSold','MoSold']
        for i in features_list:
            self.features[i] = self.features[i].astype(str)

    def defaultfill(self):
        self.features['Functional'] = self.features['Functional'].fillna('Typ') 
        self.features['Electrical'] = self.features['Electrical'].fillna("SBrkr") 
        self.features['KitchenQual'] = self.features['KitchenQual'].fillna("TA") 

    def zerofill(self):
        features_list = ['GarageYrBlt', 'GarageArea', 'GarageCars']
        for i in features_list:
            self.features[i] = self.features[i].fillna(0)       
        
    def modefill(self):
        features_list = ['Exterior1st','Exterior2nd','SaleType']
        for i in features_list:
            self.features[i] = self.features[i].fillna(self.features[i].mode()[0]) 

    def catnumfiil(self):
        self.features['LotFrontage'] = self.features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        self.features['MSZoning'] = self.features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

    def nonefill(self):
        features_list = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
        "PoolQC",'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        for i in features_list:
            self.features[i] = self.features[i].fillna("None")

    def finalfill(self):
        objects = []
        numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        numerics = []
        for i in self.features.columns:
            if self.features[i].dtype == object:
                objects.append(i)
            if self.features[i].dtype in numeric_dtypes:
                numerics.append(i)
        self.features.update(self.features[objects].fillna('None'))
        self.features.update(self.features[numerics].fillna(0))

    def mainClean(self):
        print('Cleaning ~')
        self.toCat()
        self.defaultfill()
        self.zerofill()
        self.modefill()
        self.catnumfiil()
        self.nonefill()
        self.finalfill()
        print('Cleaned !!! ')

        return self.features, self.y_train

class FeatureEngineer(object):
    def __init__(self,data,y):
        super(FeatureEngineer,self).__init__()
        self.features = data
        self.y = y 


    def newfeatures(self):
        self.features = self.features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
        self.features['YrBltAndRemod']=self.features['YearBuilt']+self.features['YearRemodAdd']
        self.features['TotalSF']=self.features['TotalBsmtSF'] + self.features['1stFlrSF'] + self.features['2ndFlrSF']

        self.features['Total_sqr_footage'] = (self.features['BsmtFinSF1'] + self.features['BsmtFinSF2'] +
                                        self.features['1stFlrSF'] + self.features['2ndFlrSF'])

        self.features['Total_Bathrooms'] = (self.features['FullBath'] + (0.5 * self.features['HalfBath']) +
                                    self.features['BsmtFullBath'] + (0.5 * self.features['BsmtHalfBath']))

        self.features['Total_porch_sf'] = (self.features['OpenPorchSF'] + self.features['3SsnPorch'] +
                                    self.features['EnclosedPorch'] + self.features['ScreenPorch'] +
                                    self.features['WoodDeckSF']) 

    def onehot(self):
        self.features['haspool'] = self.features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
        self.features['has2ndfloor'] = self.features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
        self.features['hasgarage'] = self.features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
        self.features['hasbsmt'] = self.features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
        self.features['hasfireplace'] = self.features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
        self.features = pd.get_dummies(self.features).reset_index(drop=True)

    def main(self):
        self.newfeatures()
        self.onehot()
        print('Feature Engineer Done !!')
        return self.features.iloc[:len(self.y), :],self.y,self.features.iloc[len(self.y):,:]



   

          
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def getData(datapath):
    df = pd.read_csv(datapath)
    print ("Data is loaded!")
    print (" Data: ",df.shape[0]," data instances, and ",df.shape[1],"features")

    return df

def catAndnum(df,not_feature):

    #not_feature is a list of str that is not feature

    quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
    qualitative = [f for f in df.columns if df.dtypes[f] == 'object']
    for i in not_feature:
        quantitative.remove(i)

    return quantitative,qualitative


def showMissing(df):
    sns.set_style("whitegrid")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    plt.show()

def testDistribution(df,numneric_features):
    test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
    normal = pd.DataFrame(df[numneric_features])
    normal = normal.apply(test_normality)
    print(not normal.any())

def showDistribution(df,target):

    y = df[target]
    plt.figure(1); plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=stats.johnsonsu)
    plt.figure(2); plt.title('Normal')
    sns.distplot(y, kde=False, fit=stats.norm)
    plt.figure(3); plt.title('Log Normal')
    sns.distplot(y, kde=False, fit=stats.lognorm)
    plt.show()

import sklearn
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

#replace question marks with np.nan type
def replace_question_marks(df):
    try:
        df = df.replace({'?' : np.nan})
        print("Replaced all '?' to np.nan")
    except:
        print('No question marks found')
    return df

#check whether dataset is balanced
def check_class_distribution(df):
    print('Class distributions:')
    print(df.iloc[:,-1].value_counts())
    
#PCA dimension reducetion
def dimension_reduction(x_train, x_test, upper_bound=500, n_components=50,):
    if x_train.shape[1] >= upper_bound:
        pca = PCA(n_components=n_components, random_state=33)
        pca.fit(x_train)
        x_train= pd.DataFrame(pca.transform(x_train))
        x_test = pd.DataFrame(pca.transform(x_test))
        print("Reducing dimension form %s to %s"%(x_train.shape[1],n_components))
    return x_train, x_test

#encoder for X and y string values
def encode_labels(x_train, x_test, index=None):
    label_encoder = sklearn.preprocessing.LabelEncoder()
    df = pd.concat([x_train,x_test],axis=0)
    
    #encoding y labels
    if index == -1:
        print('Encoding y label values')
        not_null_df = df[df.notnull()]
        label_encoder.fit(not_null_df)
        x_train = label_encoder.transform(x_train)
        x_test = label_encoder.transform(x_test)
    
    #encoding x features
    else:
        print('Encoding X features')
        for i,t in enumerate(df.dtypes):
            if t == 'object':
                s_df = df.iloc[:,i]
                not_null_df = s_df.loc[s_df.notnull()]
                label_encoder.fit(not_null_df)
                try:
                    x_train.iloc[:,i] = x_train.iloc[:,i].astype('float')
                except:
                    x_train.iloc[:,i] = x_train.iloc[:,i].apply(lambda x: label_encoder.transform([x])[0] if x not in [np.nan] else x)
                try:
                    x_test.iloc[:,i] = x_test.iloc[:,i].astype('float')
                except:
                    x_test.iloc[:,i] = x_test.iloc[:,i].apply(lambda x: label_encoder.transform([x])[0] if x not in [np.nan] else x) #np.nan
    return x_train, x_test

#put class colunmn at end of dataframe
def reorder_columns(dataFrame):
    cols = dataFrame.columns.tolist()
    cols = cols[1:] + cols[:1]
    return dataFrame[cols]

#impute np.nan using given strategy
def impute_value(x_train, x_test, strategy):
    if strategy == None:
        return x_train.dropna(), x_test.dropna()
    else:
        imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
        train_type_dic = dict()#keep original train data type before impute
        for i,t in enumerate(x_train.dtypes):
            if t != 'object':
                train_type_dic[i] = t
        test_type_dic = dict()#keep original test data type before impute
        for i,t in enumerate(x_test.dtypes):
            if t != 'object':
                test_type_dic[i] = t
        x_train = pd.DataFrame(imp.fit_transform(x_train))
        x_test = pd.DataFrame(imp.transform(x_test))
#         for key in train_type_dic:
#             x_train.iloc[:,key] = x_train.iloc[:,key].astype(train_type_dic[key])
#         for key in test_type_dic:
#             x_test.iloc[:,key] = x_test.iloc[:,key].astype(test_type_dic[key])
    return x_train, x_test
    
# default normalizer -> MinMaxScaelr
def normalize_data(X_train, X_test, scaler = preprocessing.MinMaxScaler()):
#     scaler = preprocessing.StandardScaler().fit(X_train)
    print('Normalized data by scaler: %s'%type(scaler))
    scaler = scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

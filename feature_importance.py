from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
# !pip install xgboost
# !pip install lightgbm
from xgboost import XGBClassifier
# import lightgbm as LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, accuracy_score, precision_score



def GetFeatureImportances(dic_models, filename="q4_XAI_data.csv", n_feats_remove=3):
    df = pd.read_csv(filename)
    X = df.iloc[:, :df.shape[1]-1]
    y = df['Loan_Status']
#   split data into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.8)
    column_name = X.columns
    print('column_name: ', column_name)
    feature_import = []
    prec_score = []
    res_dict = {'precision': [], 'accuracy': [], 'recall': []}
    for model_name, model in dic_models.items():
            new_X_train = X_train.copy()
            new_X_test = X_test.copy()
            # we train on with full feature list
            clf1 = model.fit(new_X_train, y_train)
            y_pred = model.predict(new_X_test)
            tester = model.feature_importances_
            feature_import.append(tester)
            s = np.array(tester)
            index_sorted = np.argsort(s)
            three_highest_feats_index =  index_sorted[-n_feats_remove:]
            three_highest_feats = [tester[three_highest_feats_index[i]] for i in range(3)]
              

            new_X_train.drop(new_X_train.columns[three_highest_feats_index], axis=1, inplace=True)
            new_X_test.drop(new_X_test.columns[three_highest_feats_index], axis=1, inplace=True)

            # fit new training data on model
            clf2 = model.fit(new_X_train, y_train)
            y_pred_2 = model.predict(new_X_test)
           
            res_dict['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            res_dict['precision'].append(precision_score(y_test, y_pred_2, average='weighted'))
            res_dict['accuracy'].append(accuracy_score(y_test, y_pred))
            res_dict['accuracy'].append(accuracy_score(y_test, y_pred_2))
            res_dict['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            res_dict['recall'].append(recall_score(y_test, y_pred_2, average='weighted'))
            print('res_dict: ', res_dict)
            break
        
    res_df = pd.DataFrame(res_dict, columns=res_dict.keys(), index=['XGB', 'XGB-rtop3'])
#     res_df = pd.DataFrame(res_dict, columns=res_dict.keys(), index=['XGB', 'XGB-rtop3', 'LGBM', 'LGBM-rtop3'])
    return res_df

dic_models={"XGB":XGBClassifier(random_state = 2)}

df_featImportances = GetFeatureImportances(dic_models, filename="q4_XAI_data.csv")

df_featImportances

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
import lightgbm as ltb
from sklearn.model_selection import train_test_split



from sklearn.metrics import recall_score, accuracy_score, precision_score


def GetFeatureImportances(dic_models, filename="XAI_data.csv"):
    df = pd.read_csv(filename)
    X = df.iloc[:, :df.shape[1]-1]
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.9)
    print(X.columns)
    # print(y.head())
    feature_import = []

    for model_name, model in dic_models.items():
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            tester = model.feature_importances_
            feature_import.append(tester)
            # print(len(tester))
            
            # print(y_pred)
            # break
    new_res = np.array(feature_import).T
    res_df = pd.DataFrame(new_res, columns=dic_models.keys(), index=X.columns)
    return res_df
dic_models={"ET":ExtraTreesClassifier(random_state=1),"XGB":XGBClassifier(random_state = 2),"RF":RandomForestClassifier(random_state = 2),"LGBM": ltb.LGBMClassifier(random_state = 2)}
df_featImportances = GetFeatureImportances(dic_models, filename="XAI_data.csv")

df_featImportances
import numpy as np
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

#讀取資料
A287570_col_names=['半徑5','半徑10','半徑15','半徑20','半徑25','半徑30','半徑35','半徑40','半徑45','半徑50',
           '總點數','X軸平均值','X軸標準差','X軸規格上限','X軸規格下限','X軸Ca','X軸Cp','X軸CpK','X軸3Sigma','X軸Avg+3Sigma',
           'Y軸平均值','Y軸標準差','Y軸規格上限','Y軸規格下限','Y軸Ca','Y軸Cp','Y軸CpK','Y軸3Sigma','Y軸Avg+3Sigma',
           '偏移量平均值','偏移量標準差','偏移量規格上限','偏移量規格下限','製程標準度Ca','製程精密度Cp','製程能力CpK','製程3Sigma','製程Avg+3Sigma','label','path']
A287570_stat=pd.read_csv('A287570_STAT.csv',names=A287570_col_names,header=0)
A296960_stat=pd.read_csv('A296960_STAT_path.csv',index_col=0)
A158200_stat=pd.read_csv('A158200_stat_path.csv',index_col=0)


A287570_0=A287570_stat[A287570_stat['label']==0]
A287570_1=A287570_stat[A287570_stat['label']==1]

A296960_0=A296960_stat[A296960_stat['label']==0]
A296960_1=A296960_stat[A296960_stat['label']==1]

A158200_0=A158200_stat[A158200_stat['label']==0]
A158200_1=A158200_stat[A158200_stat['label']==1]


#各個圖號抽樣出數張照片
A158200_0_frac=A158200_0.sample(9,random_state=1)
A158200_1_frac=A158200_1.sample(25,random_state=1)

A296960_1_frac=A296960_1.sample(16,random_state=1)
A296960_0_frac=A296960_0.sample(17,random_state=1)

A287570_0_frac=A287570_0.sample(14,random_state=1)
A287570_1_frac=A287570_1.sample(19,random_state=1)


frames=[A158200_0_frac,A158200_1_frac,A296960_1_frac,A296960_0_frac,A287570_0_frac,A287570_1_frac]
General_train_df=pd.concat(frames)
General_train_df=pd.DataFrame(General_train_df)
General_train_df=General_train_df.set_index(pd.Index(range(len(General_train_df))))
General_train_df=General_train_df.drop(['path'],axis=1)


#模型訓練
train_data = TabularDataset(General_train_df)

hyperparameters = {
               # 'NN': {'num_epochs': 500},
               'GBM': ['GBMLarge'],
               'CAT': {'iterations': 10000},
               'RF': {'n_estimators': 500},
               'XT': {'n_estimators': 300},
               'KNN': {},
}
predictor = TabularPredictor(label='label',eval_metric=('f1')).fit(train_data=train_data,hyperparameters=hyperparameters,num_bag_folds=5)

#預測A287570資料集
test_data=TabularDataset(A287570_stat)
A287570_test=A287570_stat.drop(['label'],axis=1)
label=A287570_stat['label']
predictions = predictor.predict(test_data)

y_predproba = predictor.predict_proba(test_data, as_multiclass=False)

A287570_test['label']=label
A287570_test['pred']=y_predproba
A287570_test['predictions']=predictions

print(predictor.evaluate(test_data, silent=True))
print(predictor.leaderboard(test_data, silent=True))


#預測A296960資料集
test_data=TabularDataset(A296960_stat)
A296960_test=A296960_stat.drop(['label'],axis=1)
label=A296960_stat['label']
predictions = predictor.predict(test_data)

y_predproba = predictor.predict_proba(test_data, as_multiclass=False)

A296960_test['label']=label
A296960_test['pred']=y_predproba
A296960_test['predictions']=predictions

print(predictor.evaluate(test_data, silent=True))
print(predictor.leaderboard(test_data, silent=True))

#預測A158200資料集
test_data=TabularDataset(A158200_stat)
A158200_test=A158200_stat.drop(['label'],axis=1)
label=A158200_stat['label']
predictions = predictor.predict(test_data)

y_predproba = predictor.predict_proba(test_data, as_multiclass=False)

A158200_test['label']=label
A158200_test['pred']=y_predproba
A158200_test['predictions']=predictions

print(predictor.evaluate(test_data, silent=True))
print(predictor.leaderboard(test_data, silent=True))
from autogluon.tabular import TabularDataset, TabularPredictor



def autogluon(data):
	#分割資料集
  X = data.drop(['label'],axis=1)
  y = data['label']
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=0,stratify=y)

  train=pd.concat([X_train,y_train],axis=1)
  test=pd.concat([X_test,y_test],axis=1)
  label=test['label']
  test=test.drop(columns=['label'])

  train_data = TabularDataset(train)
  test_data = TabularDataset(test)

  hyperparameters = {
               # 'NN': {'num_epochs': 500},
               'GBM': ['GBMLarge'],
               'CAT': {'iterations': 10000},
               'RF': {'n_estimators': 500},
               'XT': {'n_estimators': 300},
               'KNN': {},
  }
  predictor = TabularPredictor(label='label',eval_metric=('f1')).fit(train_data=train_data,hyperparameters=hyperparameters,num_bag_folds=5)


  y_predproba = predictor.predict_proba(test_data, as_multiclass=False)

  test['label']=label
  test['pred']=y_predproba

  print(predictor.evaluate(test_data, silent=True))
  print(predictor.leaderboard(test_data, silent=True))
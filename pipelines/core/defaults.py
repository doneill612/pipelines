# Should convert to derived dictionary object...
''' Default data location for Le Particulier '''
default_data_path = ''
''' Default Random Forest model hyperparameters '''
default_rf_params = {
    # hyperparameters
    'forest_params': dict(bootstrap=True, class_weight=None,
                          criterion='gini', max_depth=None,
                          max_features='sqrt', max_leaf_nodes=None,
                          min_impurity_split=1e-07, min_samples_leaf=5,
                          min_samples_split=3, min_weight_fraction_leaf=0.0,
                          n_estimators=500, n_jobs=-1,
                          oob_score=False, random_state=1337,
                          verbose=0, warm_start=False),
    # test % from a train-test split... results in 85% train 15% test
    'test_size': 0.15,
    # target column name being predicted by model
    'target': 'pnd',
    # feature columns to be used in training the model. MUST include target
    # coulumn name
    'features': []
}

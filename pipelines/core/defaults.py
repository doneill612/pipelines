# -*- coding: utf-8 -*-
import copy
from sklearn.metrics import make_scorer, recall_score, f1_score, precision_score

''' Default data location'''
default_data_path = ''

'''
Default Random Forest model hyperparameters.

The entries in this dictionary can be updated, with the format being,
>>> 'randomforestclassifier__{RandomForestClassifier parameter name here}'
For documentation regarding the RandomForestClassifier parameters that can
be used, see:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

If you plan on using a different configuration for the parameter grid other than
the one in this default dict, it would be best to create your own dict, and update
it with a deepycopy of the base_parameters dict.

See tests/pipeline_test.py for implementation details.
'''
default_rf_parameter_grid = {
    # minimum samples required to split a leaf node
    'randomforestclassifier__min_samples_split': [3, 5, 10],
    # number of estimators (trees) to use
    'randomforestclassifier__n_estimators' : [128],
    # maximum depth of each fully grown tree
    'randomforestclassifier__max_depth': [15, 25],
    # number of features to be considered when growing each tree
    'randomforestclassifier__max_features': ['sqrt', 'log2']
}

default_gbm_parameter_grid = {

}

'''
Default base parameters that should be shared amongst all model types.

When creating a parameter dict to pass to a model constructor, you can
simply update a deepcopy of this dict.

See __create_default_rf_params__().
'''
base_parameters = {
    # Scoring metrics to be used in grid search training
    'scorers': {
        'precision_score': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'f1_score': make_scorer(f1_score)
    },
    # test % from a train-test split... results in 85% train 15% test
    'test_size': 0.15,
    # target column name being predicted by model
    'target': '',
    # feature columns to be used in training the model. MUST include target
    # coulumn name
    'features': []
}

_base_params = [k for k in base_parameters.keys()]

required_rf_params = copy.deepcopy(_base_params)
required_rf_params.append('rf_parameter_grid')

required_gbm_params = copy.deepcopy(_base_params)
required_gbm_params.append('gmb_parameter_grid')

def __create_default_rf_params__():
    _default_rf_params = copy.deepcopy(base_parameters)
    _default_rf_params.update({'rf_parameter_grid': default_rf_parameter_grid})
    return _default_rf_params

def __create_default_gbm_params__():
    _default_gbm_params = copy.deepcopy(base_parameters)
    _default_rf_params.update({'rf_parameter_grid': default_gbm_parameter_grid})
    return _default_rf_params

# I very much want to move away from this implementation choice...
# Need to create generic functions that return dictionaries, and allow
# parameters to be passed.
default_rf_params = __create_default_rf_params__()
default_gbm_params = __create_default_gbm_params__()

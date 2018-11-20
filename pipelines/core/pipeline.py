# -*- coding: utf-8 -*-
import pandas as pd
from core import model_logging
from core.node import Node, NodeType
from core.abstract_model import AbstractModel
from core.nns.nn import NeuralNetwork
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import make_pipeline


class ModelPipeline(object):
    '''
    Represents a machine learning model pipeline.

    This is essentially a wrapper object around imblearn's Pipeline class.
    A model pipline is composed of a list of "nodes." Nodes are steps which
    are executed sequentially in order to normalize data, perform subsampling,
    dimensionality reduction, and eventually train a learning model.

    Every model pipeline must contain a terminal node which is the model to be
    trained. Not every model pipeline must include noramlization and/or preprocessing
    steps. In such a case, the terminal node is the sole node in the pipeline.

    This object will NOT enforce and/or verify "correct" choice of ordering
    for nodes. It is at the discretion of the user to implement good practice.
    It is also worth noting that upon adding a model node to the pipeline, the
    pipeline will "close", and not allow further addition of nodes.

    Preprocessing => Subsampling => Training / Validation => Testing.

    See tests/pipeline_test.py for an example implementation.
    '''
    def __init__(self):
        '''
        params:
           _nodes           : an ordered list of pipeline steps which will be
                              executed sequentially when `fit()` is invoked
           _terminal_node   : the model node, and last node in the _nodes collection
           _pipeline        : a `Pipeline` object
                              see imblearn.pipeline.Pipeline
           _accepting_nodes : indicates whether or not this model pipeline can
                              accept new nodes
        '''
        self._nodes = list()
        self._terminal_node = None
        self._pipeline = None
        self._accepting_nodes = True

    def add_std_scaler_node(self, node_name: str):
        '''
        Adds an average standard rescaler node to the node collection.
        Average standard scaling is a data preprocessing step in a ML pipeline,
        and performs the following operation on a feature x:

        x' = (x - mean(x)) / stddev(x)

        params:
            node_name : a string representing the name of this node
        '''
        _std_scaler_node = Node.std_scaler_node(name=node_name)
        self._add_node(_std_scaler_node)

    def add_pca_node(self, node_name: str, components: int):
        '''
        Adds a principal component analysis node to the node collection.
        PCA is a dimensionality reduction technique, and is typically
        a data preprocessing step in a ML pipeline.

        params:
            node_name  : a string representing the name of this node
            components : the number of principal components to use in PCA
        '''
        _pca_node = Node.pca_node(name=node_name, n_components=components)
        self._add_node(_pca_node)

    def add_smote_node(self, node_name: str, random_state: int=1337, ratio: float=1.0):
        '''
        Adds a node to the node collection to perform Synthetic Minority
        Oversampling (Synthetic Minority Oversampling Technique).
        SMOTE is a subsampling technique which can be used in ML pipelines
        in which a class imbalance exists on the target variable.

        params:
            node_name    : a string representing the name of this node
            random_state : a random seed to be used internally by the oversampler
            ratio        : the desired ratio between the majority class and
                           the minority class after oversampling is performed
        '''
        _smote_node = Node.smote_node(name=node_name, seed=random_state, ratio=ratio)
        self._add_node(_smote_node)

    def add_model_node(self, node_name: str, model):
        '''
        Adds a model node to the node collection. The model node object must be
        an `AbstractModel` subclass.

        If the invocation of this method succeds, the model pipeline will close
        as the model node becomes the terminal node in the sequence.

        params:
            node_name  : a string representing the name of this node
            model      : an `AbstractModel` subclass representing the machine
                         learning model to be trained
        '''
        if not isinstance(model, AbstractModel):
            model_logging.fatal('assertion', 'Attempted to add unsupported '
                                'model type. Models must be subclasses of '
                                'AbstractModel.')
        model_node = Node.dl_node(name=node_name, model=model) \
                     if isinstance(model, NeuralNetwork) \
                     else Node.ml_node(name=node_name, model=model)
        self._terminal_node = model_node
        self._build_pipeline()
        self._accepting_nodes = False
        model_logging.info('Pipeline complete with %i nodes.' % len(self._nodes))

    def fit(self):
        model_logging.info('Fitting pipeline...')
        model = self._terminal_node[1]
        if not isinstance(model, NeuralNetwork):
            # need to rework _params dictionary, individual models
            # should not have uniquely named parameter_grid entries.
            if model._params['parameter_grid'] is not None:
                self._fit_gs()
            else:
                self._fit_ml_model()
        else:
            self._fit_dl_model()

    def _fit_ml_model(self):
        # TODO
        pass


    def _transform(self, X_train, X_validate, X_test):
        model_logging.info('DL Model requires a priori transform execution.')
        model_logging.info('Running pipeline transforms...')
        X_train_transformed = X_train
        X_test_transformed = X_validate
        X_validate_transformed = X_test
        train_mod_count = 0
        validate_mod_count = 0
        test_mod_count = 0
        for node in self._nodes:
            if node.is_tranform_node:
                X_train_transformed = node.transform(X_train)
                X_validate_transformed = node.transform(X_validate)
                X_test_transformed = node.transform(X_test)
                train_mod_count += 1
                validate_mod_count += 1
                test_mod_count += 1
            elif node.is_fit_transform_node:
                X_train_transformed = node.transform(X_train)
                X_validate_transformed = node.transform(X_validate)
                train_mod_count += 1
                validate_mod_count += 1
        model_logging.info('%i transforms executed successfully.' % len(self._nodes))
        model_logging.info('Training set mod count = %i' % len(train_mod_count))
        model_logging.info('Validation set mod count = %i' % len(validate_mod_count))
        model_logging.info('Test set mod count = %i' % len(test_mod_count))
        return X_train_transformed, X_validate_transformed, X_test_transformed

    def _fit_dl_model(self):
        model_node = self._terminal_node
        model_op = model_node.get_op()
        model.compile()

        data = model_op.build_train_test_set(w_validate=True)
        X_train = data['X_train']
        X_validate = data['X_validate']
        X_test = data['X_test']
        y_train = data['y_train']
        y_validate = data['y_validate']
        y_test = data['y_test']

        X_train, X_validate, X_test = self._transform(X_train, X_validate, X_test)
        model.fit(X_train, y_train, X_validate, y_validate)
        model_logging.info('Training complete!')
        model.evaluate(X_test, y_test)



    def _fit_gs(self, folds: int=10, refit: str='f1_score', n_jobs: int=-1):
        '''
        Sequentially executes the nodes in the model pipeline.

        This method is used for grid search training.

        The model node in the pipeline contains a dictionary entry 'parameter_grid'
        which contains the model hyperparameters to be tested. The different combinations of
        hyperparameters constitute a collection of ML models to be trained.
        A grid serach is performed to find the optimal combination of model parameters
        to fit the training data. K-fold cross validation is used
        to validate each model in the grid search.

        Once training is complete, the test metrics defined in the model node
        will be evaluated on a training set and printed to the console.

        params:
            folds  : the number of splits to use in the K-fold validator
            refit  : the refitting metric to be used by the grid search
            n_jobs : number of processors to be used in the training phase.
                     -1 indicates 'use all processors'
        '''
        if self._accepting_nodes:
            model_logging.fatal('Could not fit pipeline. Model pipeline is still '
                                'accepting nodes, meaning no terminal (model) '
                                'node has yet been added.')
        model = self._terminal_node[1]
        parameter_grid = model.get_parameter_grid()
        scorers = model.get_scorers()
        skf = StratifiedKFold(n_splits=folds)
        grid_search = GridSearchCV(estimator=self._pipeline, param_grid=parameter_grid,
                                   scoring=scorers, refit=refit,
                                   cv=skf, return_train_score=True, n_jobs=n_jobs)
        X_train, X_test, y_train, y_test = model.build_train_test_set()
        grid_search.fit(X_train, y_train)
        model_logging.info('Training complete.')
        model_logging.info('Optimal parameters found:')
        model_logging.info(grid_search.best_params_)
        test_predictions = grid_search.predict(X_test)
        self._print_confusion_matrix(y_test, test_predictions)

    def _print_confusion_matrix(self, labels, predictions):
        cm = confusion_matrix(labels, predictions)
        df = pd.DataFrame(cm, columns=['Predicted 0', 'Predicted 1'],
                              index=['Actually 0', 'Actually 1'])
        print(df)

    def _build_pipeline(self):
        self._add_node(self._terminal_node)
        fit_transform_nodes = [node if node.is_fit_transform_node \
                                    or node.is_tranform_node \
                                    for node in self._nodes]
        self._pipeline = make_pipeline(*fit_transform_nodes)

    def _verify_node_name(self, node_name: str):
        for node in self._nodes:
            _node_name = node[0]
            if _node_name == node_name:
                model_logging.fatal('assertion', 'Node with name \'%s\' already '
                                    'exists in this model pipeline.' % node_name)

    def _add_node(self, node):
        if not self._accepting_nodes:
            model_logging.fatal('assertion', 'Model pipeline is no longer'
                                'accepting nodes. This is because a '
                                'model node was already added. Model nodes '
                                'are always terminal nodes in the pipeline.')
        self._verify_node_name(node.get_name())
        self._nodes.append(node)

    # def _add_node(self, node_name: str, node_op):
    #     if not self._accepting_nodes:
    #         model_logging.fatal('assertion', 'Model pipeline is no longer'
    #                             'accepting nodes. This is because a '
    #                             'model node was already added. Model nodes '
    #                             'are always terminal nodes in the pipeline.')
    #     self._verify_node_name(node_name)
    #     self._nodes.append((node_name, node_op,))

# -*- coding: utf-8 -*-
from core.pipeline import ModelPipeline
from core.defaults import default_data_path, default_rf_params
from bagging.rf import RandomForest

if __name__ == '__main__':
    pipeline = ModelPipeline()
    pipeline.add_std_scaler_node('std_scaling')
    pipeline.add_smote_node('smote', random_state=777)
    pipeline.add_model_node('rf', RandomForest(default_data_path, default_rf_params))
    pipeline.fit()

from bagging.rf import RandomForest
from core.defaults import default_data_path, default_rf_params

if __name__ == '__main__':
    rf = RandomForest(default_data_path, default_rf_params)
    rf.train()
    rf.test()

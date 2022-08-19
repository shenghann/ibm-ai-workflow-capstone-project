import os, sys
import csv
import unittest
from ast import literal_eval
import pandas as pd
from pathlib import Path

import sys
# Append parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logger

class LoggerTest(unittest.TestCase):
    """
    test the essential functionality
    """

    def test_01_train(self):
        """
        ensure log file is created
        """

        log_file = logger.LOG_FILES['train'] 
        # if os.path.exists(log_file):
        #     os.remove(log_file)

        ## update the log
        model_name = 'test'
        data_shape = (100,10)
        eval_test = {'rmse':0.5}
        runtime = "1.2"
        model_version = 0.1
        model_version_note = "test model"
        
        logger.log_train(model_name, data_shape,eval_test, runtime,
                         model_version, model_version_note)

        self.assertTrue(log_file.exists())
                

    def test_02_predict(self):
        """
        ensure log file is created
        """

        log_file = logger.LOG_FILES['pred'] 
        # if os.path.exists(log_file):
        #     os.remove(log_file)

        ## update the log
        y_pred = [0]
        runtime = "1.2"
        model_version = 0.1
        query = ['United Kingdom',2018,5,1]

        logger.log_pred(y_pred, query, runtime,
                           model_version)
        
        self.assertTrue(log_file.exists())

### Run the tests
if __name__ == '__main__':
    unittest.main()

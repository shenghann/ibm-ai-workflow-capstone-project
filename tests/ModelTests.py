import sys, os
import unittest

# Append parent directory to import path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import model, datalib

class ModelTest(unittest.TestCase):
    """
    test the essential functionality
    """
        
    def test_01_train(self):
        """
        test the train functionality
        """

        ## train the model
        model.process_and_train(test=True)
        self.assertTrue(any((model.MODEL_ROOT).glob('test*.joblib')))

    def test_02_load(self):
        """
        test the train functionality
        """
                        
        ## load the model
        load_model, _ = model.load_model('United Kingdom')
        
        self.assertTrue('predict' in dir(load_model))
        self.assertTrue('fit' in dir(load_model))

       
    def test_03_predict(self):
        """
        test the predict function input
        """

        country = 'United Kingdom'
        year = 2018
        month = 1
        day = 5
        result = model.predict_model(country, year, month, day)

        self.assertTrue(result['y_pred'][0] > 0.0)

          
### Run the tests
if __name__ == '__main__':
    unittest.main()
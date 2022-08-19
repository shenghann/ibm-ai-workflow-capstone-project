
import re
import unittest
import requests
from ast import literal_eval
import numpy as np

port = 8000

try:
    requests.get(f'http://127.0.0.1:{port}')
    server_available = True
except:
    server_available = False


class ApiTest(unittest.TestCase):

    # @unittest.skipUnless(server_available, "local server is not running")
    # def test_01_train(self):
    #     """
    #     test the train functionality
    #     """
      
    #     r = requests.post(f'http://127.0.0.1:{port}/train?mode=test')
    #     self.assertTrue('Training completed' in r['message'])
    
    @unittest.skipUnless(server_available, "local server is not running")
    def test_02_predict_empty(self):
        """
        ensure appropriate failure types
        """
    
        ## provide no data at all 
        r = requests.post(f'http://127.0.0.1:{port}/predict')
        self.assertEqual(r.status_code, 400)

        ## provide improperly formatted data
        r = requests.post(f'http://127.0.0.1:{port}/predict', json={"key":"value"})     
        self.assertEqual(r.status_code, 500)
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_03_predict(self):
        """
        test the predict functionality
        """

        payload_data = {
            'country': 'United Kingdom',
            'year': 2018,
            'month': 1,
            'day': 5,
        }

        r = requests.post(f'http://127.0.0.1:{port}/predict', json=payload_data)
        response = literal_eval(r.text)

        for p in response['y_pred']:
            self.assertTrue(p  > 0.0)

    @unittest.skipUnless(server_available, "local server is not running")
    def test_04_logs(self):
        """
        test the log functionality
        """

        type = 'train'
        r = requests.get(f'http://127.0.0.1:{port}/logs?type={type}')
        logs = literal_eval(r.text)
        
        self.assertTrue(len(logs) > 0)

### Run the tests
if __name__ == '__main__':
    unittest.main()

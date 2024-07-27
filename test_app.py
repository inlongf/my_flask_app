import unittest
from app import app

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_train(self):
        response = self.app.post('/train', json={"some": "data"})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model trained successfully", response.get_json()["message"])

    def test_predict(self):
        response = self.app.post('/predict', json={"data": [5.1, 3.5, 1.4, 0.2]})
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.get_json()["prediction"], list)

    def test_predict_invalid_data(self):
        response = self.app.post('/predict', json={"data": "invalid"})
        self.assertEqual(response.status_code, 500)

if __name__ == '__main__':
    unittest.main()

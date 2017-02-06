import os
from context import application
import unittest
import tempfile
import flask

class FlaskrTestCase(unittest.TestCase):

    def setUp(self):
        self.db_fd, application.application.config['DATABASE'] = tempfile.mkstemp()
        application.application.config['TESTING'] = True
        self.application = application.application.test_client()
        with application.application.app_context():
            application.init_db()

    def tearDown(self):
        os.close(self.db_fd)
        os.unlink(application.application.config['DATABASE'])

class TestViews(unittest.TestCase):
    def test_getting_example_data(self):
        with application.application.test_request_context( '/getExampleData?filename=samsung.csv'):
            assert flask.request.path == '/getExampleData'

if __name__ == '__main__':
    unittest.main()

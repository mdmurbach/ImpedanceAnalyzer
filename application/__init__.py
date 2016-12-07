"""
Application package for Flask
=====

Available modules
----------

views
    handles the routing of requests

fitModels
    contains functions for the fitting of both equivalent circuit and physics-based model_runs-full

"""

from flask import Flask

application = Flask(__name__, template_folder='./templates', static_url_path='/impedance-application/static')
application.config.from_object('config')
application.debug=True

from application import views

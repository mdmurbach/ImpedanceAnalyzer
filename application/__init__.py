from flask import Flask

application = Flask(__name__, template_folder='./templates', static_url_path='/impedance-application/static')
application.config.from_object('config')
application.debug=True

from application import views

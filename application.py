from __future__ import print_function
import os
from application import application

if __name__ == '__main__':
    port = int(os.environ.get('PORT',5000))
    application.run(host='0.0.0.0', port=port)

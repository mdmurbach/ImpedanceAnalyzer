Contribute to Impedance Analyzer
================================

Local Setup
------------------

Recommended minimum environment:

* `Python <https://www.python.org/>`_
* `git <https://git-scm.com/>`_
* `virtualenv <https://python-guide.readthedocs.io/en/latest/dev/virtualenvs/#virtualenv>`_

The following assumes you have all of the above tools installed already.

Windows
^^^^^^^^

1. Clone the project:
::

    > git clone https://github.com/mdmurbach/ImpedanceAnalyzer.git
    > cd ImpedanceAnalyzer

2. Create and initialize the virtual environment for the project:
::

    > virtualenv flask
    > cd flask/Scripts
    > activate
    > cd ../..
    > pip install -r requirements.txt

3. Run the python flask server
::

    > python application.py

4. Open http://localhost:5000

Flask Application Structure
---------------------------

ImpedanceAnalyzer's structure is a Flask application with the structure shown below.

.. code-block:: python

    \ImpedanceAnalyzer
        \.ebextensions      <-- setup files for executing code on EC2 instances
        \.elasticbeanstalk  <-- config files for setting up Elastic Beanstalk environment
        \application        <-- main module
            \static         <-- folder for static (data, images, js, css, etc.) files
            \templates      <-- contains html templates for pages
            __init.py__     <-- makes this folder a module
            fitModels.py    <-- python functions for fitting equivalent circuit and physics-based models
            views.py        <-- responsible for routing requests to different pages
        \docs               <-- contains files associated with this documentation
        application.py      <-- .py file for starting Flask app
        config.py           <-- config file for Flask app
        requirements.txt    <-- list of python packages used to setup environment

Flask API Backend
- main webpage: `POST` request to index.html

    - request contains:

        - a

    - response contains:

        - 'upload': False
        - 'data': example_data
        - 'ec_parameters': ec_parameters
        - 'ecFit': ecFit
        - 'p2d_parameters': p2d_parameters
        - 'p2dFit': p2dFit
        - 'p2d_residuals': p2d_residuals
        - 'p2d_simulations': p2d_simulations
        - 'p2d_names': p2d_names

At the heart of ImpedanceAnalyzer is the application package:

.. automodule:: application.fitModels
    :members:

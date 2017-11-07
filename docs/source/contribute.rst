Contribute to Impedance Analyzer
================================

Local Setup
------------------

Recommended minimum environment:

* `Python <https://www.python.org/>`_
* `git <https://git-scm.com/>`_
* `conda <https://conda.io/docs/index.html>`_

The following assumes you have all of the above tools installed already and are using the git Bash shell.

Windows
^^^^^^^^

1. Clone the project:
::

    > git clone https://github.com/mdmurbach/ImpedanceAnalyzer.git
    > cd ImpedanceAnalyzer

2. Create and initialize the virtual environment for the project:
::

    > conda env create -n impedance-analyzer-env python=3.4
    > conda install scipy=0.19.1
    > pip install -r requirements.txt

3. Use start.bat to activate the environment and start the application
::

    > ./start.bat

4. If a browser window doesn't open. Navigate to http://localhost:5000/
::


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
            fitPhysics.py   <-- python functions for fitting physics-based models
            ECfit           <-- module for fitting equivalent circuits
            views.py        <-- responsible for routing requests to different pages
        \docs               <-- contains files associated with this documentation
        application.py      <-- .py file for starting Flask app
        config.py           <-- config file for Flask app
        requirements.txt    <-- list of python packages used to setup environment

Flask API
^^^^^^^^^

The views module contains the routing structure for the flask application

.. automodule:: application.views
    :members:

Functions for Model Fitting
---------------------------
At the heart of ImpedanceAnalyzer is the ability to fit models to data:

Physics-based Models
^^^^^^^^^^^^^^^^^^^^

.. automodule:: application.fitPhysics
    :members:

Equivalent Circuit Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: application.ECfit.fitEC
    :members:

.. automodule:: application.ECfit.utilities
    :members:

Circuit elements can be added to the circuit_elements.py

.. automodule:: application.ECfit.circuit_elements
    :members:

Documentation
-------------

This project is documented using Sphinx. To rebuild the documentation:
::

    > cd docs
    > ./make.bat html

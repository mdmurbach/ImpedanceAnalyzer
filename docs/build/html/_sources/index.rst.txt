.. Impedance Analyzer documentation master file, created by
   sphinx-quickstart on Wed Dec 07 13:32:27 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========================================
Welcome to Impedance Analyzer's documentation!
===========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

==========================
Flask Application Structure
==========================

ImpedanceAnalyzer's structure is a Flask application with the structure shown below.

.. code-block:: python

    \impedance-analyzer
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


.. automodule:: application

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

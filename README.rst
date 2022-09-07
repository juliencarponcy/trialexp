========
trialexp
========


.. image:: https://img.shields.io/pypi/v/trialexp.svg
        :target: https://pypi.python.org/pypi/trialexp

.. image:: https://img.shields.io/travis/juliencarponcy/trialexp.svg
        :target: https://travis-ci.com/juliencarponcy/trialexp

.. image:: https://readthedocs.org/projects/trialexp/badge/?version=latest
        :target: https://trialexp.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Package to analyze PyControl and PyPhotometry experiments by trials, and integrate with other data such as DeepLabCut or spikes data


* Free software: MIT license
* Documentation: https://trialexp.readthedocs.io.


Features
--------

* TODO

Install
-------

* Make sure you have conda installed
* download https://github.com/juliencarponcy/trialexp 
* or, if you have git installed:
::
git clone https://github.com/juliencarponcy/trialexp
::
* Navigate into the root folder
::
cd trialexp (navigate inside the root trialexp repository)
::
* Create environment called trialexp (you can change the name of the environment by modifying the trialexp.yaml file
::
conda env create -f trialexp.yaml
::

Usage
-----

* Activate notebooks by command line using:

``conda activate trialexp``
``jupyter-notebook``
    

In the ```/notebooks``` folder can then browse the different templates notebooks, create your own notebook or copy and edit the different workflow notebooks.
  
* You can alternatively open the different workflow notebooks in a code editor which support jupyter notebooks.
  
* Or you can create a new python script or notebook and import trialexp modules


Credits
-------

This package is an extension on the work of Thomas Akam for:
- **PyControl** (Open source, Python based, behavioural experiment control)
    - pycontrol.readthedocs.io
    - github.com/pyControl/code  
      
- **PyPhotometry** (Open source, Python based, fiber photometry data acquisition)
    - github.com/pyPhotometry/code
    - pyphotometry.readthedocs.io  
      
This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.  
  
- Cookiecutter: https://github.com/audreyr/cookiecutter  
- `audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

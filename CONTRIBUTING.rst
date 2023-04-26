.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.


Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. In your PR, you should briefly explain what it is about.
2. All new PR (unless trivial or critical) will be merged into `dev` branch first. After extensive testing, the `dev` branch will be merged into the master branch regularly. No new significant functions maybe added to the `master` branch directly to ensure that the master branch is stable.
   

Get Started!
------------

Ready to contribute? Here's how to set up `trialexp` for local development.

1. Fork the `trialexp` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/trialexp.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv trialexp
    $ cd trialexp/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 trialexp tests
    $ python setup.py test or pytest
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.


Tips
----

To run a subset of tests::


    $ python -m unittest tests.test_trialexp

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.

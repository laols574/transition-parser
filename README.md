# Objectives

The learning objectives of this assignment are to:
1. practice working with dependency tree data 
2. become familiar with training transition-based parsers

# Setup your environment

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.7 or higher)](https://www.python.org/downloads/)
  **Note the new version requirement!**
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [scipy (version 1.1 or higher)](https://www.scipy.org/)
* [scikit-learn (version 0.20 or higher)](http://scikit-learn.org/)
* [pytest](https://docs.pytest.org/)

# Check out a new branch

Before you start editing any code, you will need to create a new branch in your
GitHub repository to hold your work.

1. Go to the repository that GitHub Classroom created for you,
`https://github.com/UA-LING-439-SP19/transition-parser-<your-username>`, where
`<your-username>` is your GitHub username, and
[create a branch through the GitHub interface](https://help.github.com/articles/creating-and-deleting-branches-within-your-repository/).
You may name the branch anything you like.
2. Clone the repository to your local machine and checkout the branch you
just created:
   ```
   git clone -b <branch> https://github.com/UA-LING-439-SP19/transition-parser-<your-username>.git
   ```
   where `<branch>` is whatever you named your branch.


# Write your code

You will implement two functions, `read_conllu` and `parse`, and two methods
for each of two classes, `Oracle` and `Classifier`.
A template for each of these has been provided in the `depparse.py` file.
You should read the documentation strings (docstrings) in each of methods in
that file, and implement the methods as described.
Write your code below the docstring of each method; **do not delete the
docstrings**.

The following objects and functions may come in handy:
* [collections.deque](https://docs.python.org/3/library/collections.html#collections.deque)
* [sklearn.feature_extraction.DictVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html)
* [sklearn.preprocessing.LabelEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
* [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 


# Test your code

Tests have been provided for you in the `test_depparse.py` file.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_depparse.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
...
collected 6 items                                                              

test_depparse.py FFFFEx                                                  [100%]

==================================== ERRORS ====================================
...
____________________________ test_oracle_round_trip ____________________________

    def test_oracle_round_trip():
        # take the first 50 parses from the training data
        parses = depparse.read_conllu("UD_English-EWT/en_ewt-ud-train.conllu")
>       for i, deps in enumerate(itertools.islice(parses, 50)):
E       TypeError: 'NoneType' object is not iterable

test_depparse.py:160: TypeError
================= 4 failed, 1 xfailed, 1 error in 0.52 seconds =================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
...
collected 6 items                                                              

test_depparse.py ....
75.3% accuracy on EWT development data
.X                                                  [100%]

==================== 5 passed, 1 xpassed in 301.35 seconds =====================
```

# Submit your code

As you are working on the code, you should regularly `git commit` to save your
current changes locally and `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment,
[create a pull request on GitHub](https://help.github.com/articles/creating-a-pull-request/#creating-the-pull-request).
where the "base" branch is "master", and the "compare" branch is the branch you
created at the beginning of this assignment.
Then go to the "Files changed" tab, and make sure that you have only changed
the `classify.py` file and that all your changes look as you would expect them
to.
**Do not merge the pull request.**

Your instructor will grade the code of this pull request, and provide you
feedback in the form of comments on the pull request.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.
Assignments that pass all but the last test will receive at least 80% of the
possible points.

The final test (marked with `@pytest.mark.xfail`) is optional.
If you succeed in making all tests (including this one) pass, you will receive
at least 90% of the possible points.

To get the remaining 10-20% of the points, make sure that your code is using
appropriate data structures, existing library functions are used whenever
appropriate, code duplication is minimized, variables have meaningful names,
complex pieces of code are well documented, etc.

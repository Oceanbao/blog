---
title: "PyGuide"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

# PyGuide

Topics on best practices of Python programming: testing, code quality, project organisation, packaging, etc.

- Vim + Python https://realpython.com/vim-and-python-a-match-made-in-heaven/
- best tip vim http://rayninfo.co.uk/vimtips.html?LMCL=UE5rO4
- Git Branching JS https://learngitbranching.js.org/

[toc]



# Setup Project

```shell
pipenv install black --dev
pipenv run black
pipenv install flake8 --dev

[flake8]
ignore = E203, E266, E501, W503
max-line-length = 88
max-complexity = 18
select = B,C,E,F,W,T4

pipenv run flake8

[mypy]
files=best_practices,test # dir
ignore_missing_imports=true

[tool:pytest]
testpaths=test

# .coveragerc
[run]
source = best_practices

[report]
exclude_lines =
    # Have to re-enable the standard pragma
    pragma: no cover

    # Don't complain about missing debug-only code:
    def __repr__
    if self\.debug

    # Don't complain if tests don't hit defensive assertion code:
    raise AssertionError
    raise NotImplementedError

    # Don't complain if non-runnable code isn't run:
    if 0:
    if __name__ == .__main__.:

pipenv run pytest --cov --cov-fail-under=100


# project cookiecutter
cookiecutter gh:...
git init
pipenv install -dev

```



# PyTest

> Sinlge-script app: benefit using below is that no need to turn project folder into package and able to specify file name

```python
target = __import__("my_sum.py")
sum = target.sum
```



## **How to Structure Simple Test**

1. What to test?
2. Unit test or integration test?

Then:

1. Create inputs
2. Run code capturing output
3. Compare with expected

Example sum(): many behaviours could be checked:

- can it sum a list of int?
- can it sum a tuple or set?
- can it sum a list of float/
- Check with bad input, e.g. a single int or string
- check negative value



## **Differences Testing Flask: Using `unittest`**

> Flask requires that the app be imported and then set in test mode. Init test client and use test client to make requests to any routes in app
>
> All of test client init is done in the `setup` method of test case

```python
# my_app is name of app

import my_app
import unittest

class MyTestCase(unittest.TestCase):
    
    def setUp(self):
        my_app.app.testing = True
        self.app = my_app.app.test_client()
        
	def test_home(self):
        result = self.app.get('/')
        # Make assertions...

# run with `python -m unittest discover`
```



## **Advanced Input: instance of class or context**

**Input is FIXTURE: create and reuse**

- Varied input but same result - **parameterisation**

- Catching errors



## **Isolating Behaviour and State** or **Side Effects**

- Refactoring code to follow Single Responsibility Principle !!
- Mocking out any method or function calls to remove side effects
- Using integration testing instead of unit testing for this piece of app



## **Integration Test**

- calling HTTP REST API
- calling Python API
- calling web service
- running a CLI

Most can follow Unit Test **Input, Run, Assert** pattern with **DIFF** that this is checking more components AT ONCE and hence having more side effects, requiring more fixtures, like DB, network socket, config file.

> Run once before deployment rather than per commit !!



**Repo Structure**

```shell
project/
|
|---my_app/
|	|---__init__.py
|---tests/
	|---unit/
		|---__init__.py
		|---test_sum.py
    |---integration/
    	|
    	|---fixtures/
    		|---test_basic.json
    		|---test_complex.json
    	|
    	|---__init__.py
    	|---test_integration.py
```



## **Automation: CI/CD - run tests, compile and publish and deploy**

- Travis CI one of many CI

- Travis CI is free for open-source projects on GitHub and GitLab

  - config `.travis.yml` 

  - ```yaml
    language: python
    python:
    	- "2.7"
    	- "3.7"
    install:
    	- pip install -r requirements.txt
    script:
    	- python -m unittest discover
    ```

    - Test against 2.7 and 3.7
    - Install all pkgs in file
    - run test

  - Once committed and pushed this file, Travis CI will run these commands every time pushing to remote Git repo and available metrics online



## **Tips**

- Linters - Tox and Travis CI have config for a test cmd

  - can provide many cmds in pipeline such as linter - commenting on code

- Passive Linting **flake8**

  - PEP 8 specs - `flake8 test.py`

- Aggressive Linting **black**

- Keeping test code clean

  - test fixtures and funcs are a great way to produce good code
    - `flake8 --max-line-length=120 tests/`

- Testing for Speed Degradation Between Changes

  - ```python
    def test():
        # ... your code
        
    if __name__ == '__main__':
        import timeit
        print(timeit.timeit("test()", setup="from __main__ import test", number=100))
    ```

  - Or **pytest** using **pytest-benchmark** plugin 

  - ```python
    def test_my_function(benchmark):
        result = benchmark(test)
    ```

- Security
  - `pip install bandit`
  - `bandit -r my_sum`



# Test-Driven Dev with PyTest

TODO



## **MOCKs**

Adding outside func call to app:

```python
def initial_transform(data):
    """
    Flatten nested dicts
    """
    for item in list(data):
        if type(data[item]) is dict:
            for key in data[item]:
                data[key] = data[item][key]
			data.pop(item)
	
    outside_module.do_sth()
    return data
```



- do not want to make LIVE call (API) to externals so instead do MOCKING

- Setting up mocks in fixtures since it's a part of test setup and can keep all setup code in one

- ```python
  @pytest.fixture(params=['nodict', 'dict'])
  def generate_initial_transform_parameters(request, mocker):
      [...]
      mocker.patch.object(outside_module, 'do_sth')
      mocker.do_sth.return_value(1)
      [...]
  ```

- Now each `initial_transform` call then `do_sth` call will be intercepted and return 1

- control fixture params to determine what mock returns - important for code branch determined by the result of outside call

- neat trick is `side_effect` allowing mocking various returns for successive calls to the same func:

- ```python
  def initial_transform(data):
      ....
      
      outside_module.do_sth()
      outside_module.do_sth()
      return data
  ```

- then set up mock so:

- ```python
  @pytest.ficture(...)
  def ...
  	mocker.patch....
      mocker.do_sth.side_effect([1, 2])
      ...
  ```

- Advanced: [Mock Server to test 3P APIs](https://realpython.com/testing-third-party-apis-with-mock-servers/)



# CI/CD

Multi-coder coordination management.

**CI** is practice of frequently building and testing each change done to code automatically and as early as possible. 

> Martin Fowler: CI is a software dev practice where members of a team integrate their work frequently, usually each person integrates at least daily - leading to multiple integrations per day. Each is verified by an automated build (and test) to detect integration errors asap.



Skipping Git repo, code, Unit Tests, 

**Unit Test**

- `flake8` for styling and errors
- `pytest` for unit testing
- **Code Coverage** is % "covered" by tests `pytest-cov` extension
- `pip install flake8 pytest pytest-cov`
- `pip freeze > requirements.txt`
- `flake8 --statistics`
- `pytest -v --cov`

Done and push to master `git add ..., git commit ... git push`

## Connect to Server CI - CircleCI

- require config `.circleci` folder within repo 

- `.yml` file uses data serialization lang, YAML, 

  - key-value, list, scalar

  - indent for structure, colons split key-value, dashes creating list

  - ```yaml
    # Python CircleCI 2.0 configuration file
    # simple exmple of:
    # 1. checking out the repo
    # 2. Installing the deps in venv
    # 3. Running linter and tests while inside the venv
    
    version: 2
    jobs: # Jobs represent a single execution of the build and are defined by a collection of steps. If you have only one job, it must be called build.
    	build: # As mentioned before, build is the name of your job. You can have multiple jobs, in which case they need to have unique names.
    		docker: # The steps of a job occur in an environment called an executor. The common executor in CircleCI is a Docker container. It is a cloud-hosted execution environment but other options exist, like a macOS environment.
    			- image: circleci/python:3.7
    		# Your repository has to be checked out somewhere on the build server. The working directory represents the file path where the repository will be stored.
    		working_directory: ~/repo
    		
    		steps:
    			# Step 1: obtian repo from GitHub
    			- checkout # The first step the server needs to do is check the source code out to the working directory. This is performed by a special step called checkout.
    			# Step 2: create venv and env setup
    			- run:
    				name: install dependencies
    				command: |
    					python 3 -m venv venv
    					. venv/bin/activate
    					pip install -r requirements.txt
    			# Step 3: run linter and tests
    			- run:
    				name: run tests
    				command: |
    					. venv/bin/activate
    					flake8 --exclude=venv* --statistics
    					pytest -v --cov=calculator
    ```

  - log in site and Add Projects -> Set Up Project > Python -> skip config.yml -> Start builiding

  - Now each **push to master branch** a job will be triggered - Jobs in sidebar for log

- Add new feature and failing test first is **TDD** - **writing test first and add code**
- Notification of failure
  - email for each failed build
  - failure notification Slack
  - on a dashboard



## Advanced Topics

- Git Workflow
  - e.g. branching strategies **peer review** to ask for peer review before merging to master
- Dependency Management and VENV
  - e.g. **Pipenv** 
- Testing
  - see elsewhere here
- Packaging
  - elsewhere
- CI
  - common for final step to leave **deployable artifact** representing a finished, packaged unit of work ready to deploy
- CD
  - ext of CI, deployment to production to minimise lead time from code to end usage



## Other CI Services

- remote and self-hosted
- **Jenkins** self-hosted
- **Travis CI, CodeShip, Semaphore, etc**
- 

# OFF: Scripting Tips

\# Highlight Tools

-   f-strings
-   pathlib
-   click
-   asyncio
-   attrs
-   streamz

## f-strings

-   easy accessing ENV variables via f''

```python
from subprocess import call
call(f'curl -s -X GET http://{URL}/v2/{IMAGE}/tags/list')
```

## pathlib

Without pathlib

```python
import glob
import os
import shutil

for file_name in glob.glob('*.txt'):
    new_path = os.path.join('archive', file_name)
    shutil.move(file_name, new_path)
```

Examples

```bash
>>> import pathlib
>>> pathlib.Path.cwd()
PosixPath('/home/gahjelle/realpython/')

>>> pathlib.Path(r'C:\Users\gahjelle\realpython\file.txt')
WindowsPath('C:/Users/gahjelle/realpython/file.txt')

>>> pathlib.Path.home() / 'python' / 'scripts' / 'test.py'
PosixPath('/home/gahjelle/python/scripts/test.py')

>>> pathlib.Path.home().joinpath('python', 'scripts', 'test.py')
PosixPath('/home/gahjelle/python/scripts/test.py')

# CREATE FILE

path = pathlib.Path.cwd() / 'test.md'
with open(path, mode='r') as fid:
    headers = [line.strip() for line in fid if line.startswith('#')]
print('\n'.join(headers))

# READ
>>> pathlib.Path('test.md').read_text()
<the contents of the test.md-file>

# COMPONENTS
>>> path
PosixPath('/home/gahjelle/realpython/test.md')
>>> path.name
'test.md'
>>> path.stem
'test'
>>> path.suffix
'.md'
>>> path.parent
PosixPath('/home/gahjelle/realpython')
>>> path.parent.parent
PosixPath('/home/gahjelle')
>>> path.anchor
'/'

>>> path.parent.parent / ('new' + path.suffix)
PosixPath('/home/gahjelle/new.md')

>>> path
PosixPath('/home/gahjelle/realpython/test001.txt')
>>> path.with_suffix('.py')
PosixPath('/home/gahjelle/realpython/test001.py')
>>> path.replace(path.with_suffix('.py'))


>>> import collections
>>> collections.Counter(p.suffix for p in pathlib.Path.cwd().iterdir())
Counter({'.md': 2, '.txt': 4, '.pdf': 2, '.py': 1})



```





```python
from pathlib import Path
p = Path('.')
p

str(p)
str(p.absolute())

p = p.absolute()
p.as_posix()

p.as_uri()

p.parent

p.relative_to(p.parent)

# example 2
q = p / 'newdir'
q

p.exists()

q.exists()

p.is_dir()

p.is_file()

# subdirectory
[x for x in p.iterdir() if x.is_dir()]

# files
[x for x in p.iterdir() if x.is_file()]

# find recursively
list(p.rglob('*'))

# directory
q.exists()

q.mkdir()
	
q.exists()

# files
fp = n / 'newfile.txt'
fp

with fp.open('wt') as f:
    f.write('The quick brown fox jumped over the lazy dog.')

fp.exists() and fp.is_file()

fp.read_text()

# removal
fp.unlink()

fp.exists()

q.rmdir()

q.exists()
```



## click

-   auto --help docs
-   params validation
-   arbitrary nesting of cmds
-   support lazy loading of subcmds at RT

```python
import click

@click.command()
@click.option('--count', default=1, help='Number of greetings')
@click.option('--name', prompt='Your name', help='The person to greet')
def hello(count, name):
  """Simple app greeting NAME for a total of COUNT times"""
  for x in range(count):
    click.echo('Hello {name}!')
if __name__ == '__main__':
  hello()
```

```bash
python app.py --help

python examples/greet.py

python examples/greet.py --name 'PyconZA 2017' --count 3
```

**setuptool integration**
check docs
or google numismatic setup github

## Example - crawl bitcoin data [asyncio, attrs, streamz, websocks]

### streamz

```python
from streamz import Stream source = Stream()
source.emit('hello') source.emit('world')


printer = source.map(print) for event in ['hello', 'world']: source.emit(event) 
L = [] 
collector = source.map(L.append) j for event in ['hello', 'world']: source.emit(event)
```



### attr (better named tuples)

```python
@attr.s(slots=True)
class Heartbeat:
    exchange = attr.ib()
    symbol = attr.ib()
    timestamp = attr.ib(default=attr.Factory(time.time))
```

used as

```python
from numismatic.events import Heartbeat
Heartbeat('bitfinex', 'BTCUSD')

import attr
attr.asdict(Heartbeat('bitfinex', 'BTCUSD'))
```



### numismatic

```python
# set up stream

from streamz import Stream
source = Stream()
printer = source.map(print)
L = []
collector = source.map(L.append)

# prep connection
from numismatic.exchanges import BitfinexExchange
bfx = BitfinexExchange(source)
subscription = bfx.listen('BTCUSD', 'trades')

# run event loop
import asyncio
loop = asyncio.get_event_loop()
future = asyncio.wait([subscription], timeout=10)
loop.run_until_complete(future)
```



try:
git clone https://github.com/snth/numismatic.git
cd numismatic
pip install -e .

**run sans args for help**
coin

**streamz as test tool before deploying KAFKA, FLINK**



# Style and Quality

## Google Style

### Exceptions

- pros: control flow of normal ops code not cluttered by error-handling code

- cons - may cause confusing miss errors

- decision - must conditions

  - Raise exceptions like: `raise MyError('Error message') or raise MyError()`

  - make use of built-in classes

  - do not use `assert` for validating argument values of a public API - assert is used to ensure **internal correctness**, not to enfore correct usage nor to indicate that some unexected event occurred - use raise 

  - Example

    - ```python
      # YES
      def connect(self, min):
          """Connects to next available port.
          
          Args:
          	min: 
          Returns:
          	new mim
          Raises:
          	ConnectionError: If no available port is found.
          """
          if min < 1024:
              # Note this raising of ValueError is not documented
              # because it is not appropriate to guarantee this specific behvaiour
              # reaction to API miuse.
              raise ValueError('Min port must be at least 1024, not {:d}.'.format(mim))
      	port = self._...
          if not port:
              raise ConnectionError('Could not connect to service on {:d} or higher'...)
      	assert port >= min, 'Unexpected port {:d} when min was {:d}'.format(port, min)
          return port
      
      # NO
      def ...:
          """
          No mention of Raises!
          """
          # no implementation of raise ConnectionError
          assert min >= 1024, ...
          port = ..
          assert port is not None
          return port
      ```

    - Lib/pkg may define their own exceptions, when doing so they must inherit from an exception class should end  with erro (foo.FooError)

    - never use catch-all `except` or catch `Exception, StandardError` unless you are 

      - re-raising the exception or
      - creating an isolation point in program where exceptions are not propagated but recorded suppresed instead, such as protecting a thread from crashing by guarding its outermost block
      - python very tolerant in this and `except` will really catch everything inclu misspelled names, `sys.exti()` calls, Ctrl-C, unittest and all kinds

    - Min the amount of code in `try/except` block, large the try, the more likely that an exception will be raised unexpectedly - hiding real error

    - use `finally` to exec code whether or not an exception is raised in `try`, often useful for cleanip, i.e. closing file

    - when capturing an exception, use `as` rather than comma

      - ```python
        try:
            raise Error()
        except Error as error:
            pass
        ```

### Global Variables

- avoid
- see Naming

### Nested/Local/Inner Classes and Functions

- fine when used to close over a local variable, inner classes are fine
- pros - allow objects limited scope, very ADT-y, commonly used for decorators
- cons - cannot be pickled !! cannot be directly tested !!legibility cost
- verdict - avoid except when closing a local value, DO NOT nest a func just to hide it from users of a module !! _predix is prefer so that it can still be accessed by tests

### Comprehensions & Generator Expressions

- concise, efficient container/iterator creation over `map, filter, lambda`
- pros - simple can be clear than dict, list, set creation, generators very efficient for avoiding list entirely
- cons - complicated ones hard to read
- verdict - each portion must fit one-line `for` clause, filter expre, multiple for not permitted

```python
# YES
result = [mapping_expr for value in iterable if filter_expr]
result = [{'key': value} for value in iterable
	if a_long_filter_expression(value)]
result = [complicated_transform(x)
	for x in iterable if predicate(x)]
descriptive_name = [
	transform({'key': key, 'value': value}, color='black')
	for key, value in generate_iterable(some_input)
	if complicated_condition_is_met(key, value)
]
result = []
for x in range(10):
	for y in range(5):
		if x * y > 10:
			result.append((x, y))
return {x: complicated_transform(x)
	for x in long_generator_function(parameter)
	if x is not None}
squares_generator = (x**2 for x in range(10))
unique_names = {user.name for user in users if user is not None}
eat(jelly_bean for jelly_bean in jelly_beans
	if jelly_bean.color == 'black')

# NO
result = [complicated_transform(
	x, some_argument=x+1)
	for x in iterable if predicate(x)]
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
return ((x, y, z)
	for x in xrange(5)
	for y in xrange(5)
	if x != y
	for z in xrange(5)
        if y != z)

```

### Default Iterators and Operators

- pros - simple and efficient, expressive, generic
- cons - cannot tell type
- verdict - use default for types support them, list, dict, files, prefer 

```python
# Yes
for key in adict: ...
if key not in adict: ...
for line in afile: ...
for k, v in adict.items(): ...
# No
for key in adict.keys(): ...
if not adict.has_key(key):...
for line in afile.readlines(): ...
for k, v in dict.iteritems(): ....
```

### Properties

```python
# Yes
class Square(object):
    """A square with two properties: a writable area and a read-only perimeter.
    To use:
    >>> sq = Square(3)
    >>> sq.area
    9
    >>> sq.perimeter
    12
    >>> sq.area = 16
    >>> sq.side
    4
    >>> sq.perimeter
    16
    """
    def __init__(self, side):
    	self.side = side
    @property
    def area(self):
    	"""Area of the square."""
    	return self._get_area()
    @area.setter
    def area(self, area):
    	return self._set_area(area)
    def _get_area(self):
    	"""Indirect accessor to calculate the 'area' property."""
        return self.side ** 2
    def _set_area(self, area):
        """Indirect setter to set the 'area' property."""
   		self.side = math.sqrt(area)
    @property
    def perimeter(self):
    	return self.side * 4
```

### Conditional !!!

```python
# Yes
if not users:
    print(' no users')
if foo == 0:
    self.handle_zero()
if i % 10 == 0:
    self.handle_multiple_of_ten()
def f(x=None):
    if x is None:
        x = []

# No
if len(users) == 0:
    ...
if foo is not None and not foo:
    ...
if not i % 0:
    ...
def f(x=None):
    x = x or []
```

### Decorators

```python
class C:
    @my_decorator
    def method(self):
        # method body
        
# equivalent to
class C:
    def method(self):
        # body
	method = my_decorator(method)
```

- use judiciouly - follow same import and naming as functions !!! 
- write test for decorators
- avoid external dependencies in decorator itself (don't rely on files, sockets, db connections, etc) 
- never use @staticmethod unless forced to in order to integrate with API defined in an existing lib, write a module level function instead !!!
- @classmethod only when writing a named constructor or class-specific routine that modidies necessary global state such as process-wide cache

### Threading

- do not rely on atomicity of built-in types
- use Queue's Queue data type as preferred way to in-proc communication. else, use threading module and its locking primitives

### Power Features

- avoid !
- metaclass, access to bytecode, on-the-fly compilation, dynamic inheritance, object reparenting, import hacks, reflection (getattr()), modificaiton of system internals etc
- -very tempting, harder to read, digest, debug
- 

## Numpy Style Docstrings

```python
# -*- coding: utf-8 -*-
"""Example NumPy style docstrings.

This module demonstrates documentation as specified by the `NumPy
Documentation HOWTO`_. Docstrings may extend over multiple lines. Sections
are created with a section header followed by an underline of equal length.

Example
-------
Examples can be given using either the ``Example`` or ``Examples``
sections. Sections support any reStructuredText formatting, including
literal blocks::

    $ python example_numpy.py


Section breaks are created with two blank lines. Section breaks are also
implicitly created anytime a new section starts. Section bodies *may* be
indented:

Notes
-----
    This is an example of an indented section. It's like any other section,
    but the body is indented to help it stand out from surrounding text.

If a section is indented, then a section break is created by
resuming unindented text.

Attributes
----------
module_level_variable1 : int
    Module level variables may be documented in either the ``Attributes``
    section of the module docstring, or in an inline docstring immediately
    following the variable.

    Either form is acceptable, but the two should not be mixed. Choose
    one convention to document module level variables and be consistent
    with it.


.. _NumPy Documentation HOWTO:
   https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

"""

module_level_variable1 = 12345

module_level_variable2 = 98765
"""int: Module level variable documented inline.

The docstring may span multiple lines. The type may optionally be specified
on the first line, separated by a colon.
"""


def function_with_types_in_docstring(param1, param2):
    """Example function with types documented in the docstring.

    `PEP 484`_ type annotations are supported. If attribute, parameter, and
    return types are annotated according to `PEP 484`_, they do not need to be
    included in the docstring:

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : str
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.

    .. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/

    """


def function_with_pep484_type_annotations(param1: int, param2: str) -> bool:
    """Example function with PEP 484 type annotations.

    The return type must be duplicated in the docstring to comply
    with the NumPy docstring style.

    Parameters
    ----------
    param1
        The first parameter.
    param2
        The second parameter.

    Returns
    -------
    bool
        True if successful, False otherwise.

    """


def module_level_function(param1, param2=None, *args, **kwargs):
    """This is an example of a module level function.

    Function parameters should be documented in the ``Parameters`` section.
    The name of each parameter is required. The type and description of each
    parameter is optional, but should be included if not obvious.

    If \*args or \*\*kwargs are accepted,
    they should be listed as ``*args`` and ``**kwargs``.

    The format for a parameter is::

        name : type
            description

            The description may span multiple lines. Following lines
            should be indented to match the first line of the description.
            The ": type" is optional.

            Multiple paragraphs are supported in parameter
            descriptions.

    Parameters
    ----------
    param1 : int
        The first parameter.
    param2 : :obj:`str`, optional
        The second parameter.
    *args
        Variable length argument list.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    bool
        True if successful, False otherwise.

        The return type is not optional. The ``Returns`` section may span
        multiple lines and paragraphs. Following lines should be indented to
        match the first line of the description.

        The ``Returns`` section supports any reStructuredText formatting,
        including literal blocks::

            {
                'param1': param1,
                'param2': param2
            }

    Raises
    ------
    AttributeError
        The ``Raises`` section is a list of all exceptions
        that are relevant to the interface.
    ValueError
        If `param2` is equal to `param1`.

    """
    if param1 == param2:
        raise ValueError('param1 may not be equal to param2')
    return True


def example_generator(n):
    """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Parameters
    ----------
    n : int
        The upper limit of the range to generate, from 0 to `n` - 1.

    Yields
    ------
    int
        The next number in the range of 0 to `n` - 1.

    Examples
    --------
    Examples should be written in doctest format, and should illustrate how
    to use the function.

    >>> print([i for i in example_generator(4)])
    [0, 1, 2, 3]

    """
    for i in range(n):
        yield i


class ExampleError(Exception):
    """Exceptions are documented in the same way as classes.

    The __init__ method may be documented in either the class level
    docstring, or as a docstring on the __init__ method itself.

    Either form is acceptable, but the two should not be mixed. Choose one
    convention to document the __init__ method and be consistent with it.

    Note
    ----
    Do not include the `self` parameter in the ``Parameters`` section.

    Parameters
    ----------
    msg : str
        Human readable string describing the exception.
    code : :obj:`int`, optional
        Numeric error code.

    Attributes
    ----------
    msg : str
        Human readable string describing the exception.
    code : int
        Numeric error code.

    """

    def __init__(self, msg, code):
        self.msg = msg
        self.code = code


class ExampleClass(object):
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they may be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section. Alternatively, attributes may be documented
    inline with the attribute's declaration (see __init__ method below).

    Properties created with the ``@property`` decorator should be documented
    in the property's getter method.

    Attributes
    ----------
    attr1 : str
        Description of `attr1`.
    attr2 : :obj:`int`, optional
        Description of `attr2`.

    """

    def __init__(self, param1, param2, param3):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1 : str
            Description of `param1`.
        param2 : :obj:`list` of :obj:`str`
            Description of `param2`. Multiple
            lines are supported.
        param3 : :obj:`int`, optional
            Description of `param3`.

        """
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = param3  #: Doc comment *inline* with attribute

        #: list of str: Doc comment *before* attribute, with type specified
        self.attr4 = ["attr4"]

        self.attr5 = None
        """str: Docstring *after* attribute, with type specified."""

    @property
    def readonly_property(self):
        """str: Properties should be documented in their getter method."""
        return "readonly_property"

    @property
    def readwrite_property(self):
        """:obj:`list` of :obj:`str`: Properties with both a getter and setter
        should only be documented in their getter method.

        If the setter method contains notable behavior, it should be
        mentioned here.
        """
        return ["readwrite_property"]

    @readwrite_property.setter
    def readwrite_property(self, value):
        value

    def example_method(self, param1, param2):
        """Class methods are similar to regular functions.

        Note
        ----
        Do not include the `self` parameter in the ``Parameters`` section.

        Parameters
        ----------
        param1
            The first parameter.
        param2
            The second parameter.

        Returns
        -------
        bool
            True if successful, False otherwise.

        """
        return True

    def __special__(self):
        """By default special members with docstrings are not included.

        Special members are any methods or attributes that start with and
        end with a double underscore. Any special member with a docstring
        will be included in the output, if
        ``napoleon_include_special_with_doc`` is set to True.

        This behavior can be enabled by changing the following setting in
        Sphinx's conf.py::

            napoleon_include_special_with_doc = True

        """
        pass

    def __special_without_docstring__(self):
        pass

    def _private(self):
        """By default private members are not included.

        Private members are any methods or attributes that start with an
        underscore and are *not* special. By default they are not included
        in the output.

        This behavior can be changed such that private members *are* included
        by changing the following setting in Sphinx's conf.py::

            napoleon_include_private_with_doc = True

        """
        pass

    def _private_without_docstring(self):
        pass
```



## Type Hinting (mypy)

```python
# inline types
>>> name: str = "Guido"
>>> pi: float = 3.142
>>> centered: bool = False
>>> names: list = ["Guido", "Jukka", "Ivan"]
>>> version: tuple = (3, 7, 1)
>>> options: dict = {"centered": False, "capitalize": True}

# should use special types - composite types
from typing import Dict, List, Tuple
names: List[str] = ["..."]
version: Tuple[int, int, int] = (3, 7, 1)
options: Dict[str, bool] = ...

# more composite types: Counter, Deque, FrozenSet, NamedTuple, Set
def create_deck(shuffle: bool = False) -> List[Tuple[str, str]]:
    ...
   
# if care not the content of sequence, use typing.Sequence
def square(elems: Sequence[float]) -> List[float]:
    return [x**2 for x in elems]

# aliasing (save repeating certain types)
Card = Tuple[str, str]
Deck = List[Card]
def deal_hands(deck: Deck) -> Tuple[Deck, Deck, Deck, Deck]:
     return (deck[0::4], ...)
    
# TypeVar - special var taking on any type, conditioned
Choosable = TypeVar("Chooseable")
def choose(items: Sequence[Choosable]) -> Choosable: 
    return random.choice(items)
reveal_type(choose(["Python"], 3, 7)) # does its best to accommodate
# now with constrain
Choosable = TypeVar("Choosable", str, float)

# protocol specifies one ore more methods that must be implemented
# e.g. all cls defining .__len__() fulfull the typing.Size protocol
# therefore annotate len() as :
from typing import Sized
def len(obj: Sized) -> int:
    return obj.__len__()
# other protocols defined include Container, Iterable, Awaitable, ContextManager
# custom protocol
from typing_extensions import Protocol
class Sized(Protocol):
    def __len__(self) -> int: ...
def len(obj: Sized) -> int:
    return obj.__len__()

# Optional for None/other argument case
def player_order(
	names: Sequence[str], start: Optional[str] = None
) -> Sequence[str]:
    ...
# hence a var either has the type specified or is None 
# equivalent: Union[None, str]
# NOTE: when using Optional or Union, must take care of correct type!
# by testing var is type !! else causes static type errors and runtime errors!!
def player_order(
	... = None
) -> Sequence[str]:
    start_idx = names.index(start)
    return names[start_idx:] + names[:start_idx]
# mypy code.py >>> Arg 1 to "index" of "list" has incompatible type
# "Optional[str]"; expected "str"
# start: str = None is ok and often default hanlded! 

# Class types has special cases such as internal returns
class Deck:
    @classmethod
    def create(cls, shuffle: bool = False) -> "Deck":
        cards = ...
        ...
        return cls(cards)
# with __future__ import annotations, can use Deck instead of "Deck" 
# EVEN BEFORE Deck is defined !!! `def create(...) -> Deck:`

# Returning self or cls
# typically not annotate self/cls
# ONE case might be needed! Superclass inheritance returning cls/self
from datetime import date
class Animal:
    def __init__(self, name: str, birthday: date) -> None:
        self.name = name
        self.birthday = birthday
        
	@classmethod
    def newborn(cls, name: str) -> "Animal":
        return cls(name, date.today())
    def twin(self, name: str) -> "Animal":
        cls = self.__class__
        return cls(name, self.birthday)
    
class Dog(Animal):
    def bark(self) -> None:
        print(f"{self.name} says woof!")
fido = Dog.newborn("Fido")
pluto = fido.twin("Pluto")
fido.bark()
pluto.bark()
# mypy dogs.py >>> error: "Animal" has no attribute "bark" x2
# ISSUE is that even though inherited Dog.newborn() and Dog.twin() methods will
# return a Dog the annotation says taht they return an Animal
# Hence be careful to ensure annotation is correct!!!
# return type should match type of self/cls by tracking
from typing import Type, TypeVar
TAnimal = TypeVar("TAnimal", bound="Animal")
class Animal:
    def __init__(...) -> None:
        ...
    @classmethod
    def newborn(cls: Type[TAnimal], name: str) -> TAnimal:
        ...
	def twin(self: TAnimal, name: str) -> TAnimal:
        ...
# TAnimal (TypeVar) used to denote return values might be inheritances of subcls of Animal
# Animal is an upper bound for TAnimal, meaning only be Animal or one of its subcls
# typing.Type[] is the typing equivalent of type() - noting that cls method expects a class
# and returns an instance of that cls

# Funcs and Methods
from typing import Callable
def do_twice(func: Callable[[str], str], arg: str) -> str:
    ...
    
    
# EXAMPLE
from typing import Tuple, Iterable, Dict, List, DefaultDict
from collections import defaultdict

def create_tree(tuples: Iterable[Tuple[int, int]]) -> DefaultDict[int, List[int]]:
    """
    Return a tree given tuples of (child, father)

    The tree structure is as follows:

        tree = {node_1: [node_2, node_3], 
                node_2: [node_4, node_5, node_6],
                node_6: [node_7, node_8]}
    """
    tree = defaultdict(list) 
    for child, father in tuples:
        if father:
            tree[father].append(child)
    return tree

print(create_tree([(2.0,1.0), (3.0,1.0), (4.0,3.0), (1.0,6.0)]))
# will print
# defaultdict( 'list'="">, {1.0: [2.0, 3.0], 3.0: [4.0], 6.0: [1.0]}

# error: tree = defaltdict(list)
# Mypy cannot infer tree is indeed of type intended
tree: DefaultDict[int, List[int]] = defaultdict(list) # inline new var definition

```

- https://kite.com/blog/python/type-hinting/
- https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html

# Package

```bash
- README.rst
- LICENSE
- setup.py	# pkg and dist management
- requirements.txt 	# optional inside setup.py
- sample	# main app
	- __init__.py
	- core.py
	- helpers.py
- docs
	- conf.py
	- index.rst
- tests
	- test_basic.py
	- test_advanced.py
```



**Makefile**

```makefile
init:
pip install -r requirements.txt

test:
py.test tests
```



### PIPENV - Python new packaging tool

- solving issues with pip, virtualenv, and good old requirements.txt
- streamline process

**Dependency MGT with requirements.txt**

- no version control
- okey, there's `flaks==0.12.1`
- BUT keep in mind that flask itself has depens as well (which pip installs automatically) BUT flask doesn't specify exact versions for its depens 
- So when flask's depens updated, the above flask version installed BUT also the latest, buggy version of its depens
- issue: **BUILD IS NOT DETERMINISTIC**
  - typical solution `pip freeze` 
  - BUT it leads to a whole new set of problems!!
  - now exact versions of all pkgs specified, you are responsible for keeping them up to date!! even they're sub-depens of flask  - what if there's a security hole discovered in Werkzeug==0.14.1 that the keeper at once patched in new .2? YOU NEED to update that version to .2 to avoid any security issues

> QUESTION: **HOW TO ALLOW FOR DETEMINSTIC BUILDS FOR PYTHON PROJECT WITHOUT GAINING THE RESPONSIBILITY OF UPDATING VERSIONS OF SUB-DEPENS?**

**Environment Control by virutalenv or venv in P3**

- depens solution -> ranging versions `package_c>=1.0,<=2.0`
- BUT still keeping updated with changes remains!!



### Pipenv Intro

- `pip install pipenv`

- two new files `Pipfile` replacing requirements.txt and `Pipfile.lock` enabling deterministic builds

- e.g. First spawn a shell in a venv to isoluate the dev of this app

  - `pipenv shell [--python 3.6]` creating new venv in default loc
  - `pipenv install flask==0.12.1`
    - `Adding flask==0.12.1 to Pipfile's [packages]... Pipfile.lock not found, creating...`
    - `pipenv install -e git+https://github.com/requests/requests.git#egg=requests`
    - also unit tests for not production but dev 
    - `pipenv install pytest --dev`
      - `--dev` put the depen in a special [dev-packages] loc in Pipfil which only get installed if --dev 
    - now ready to push to PROD, lock env `pipenv lock`
      - create/update Pipefile.lock which never needed to be edited manually
    - Now in PROD env with code and lock `pipenv install --ignore-pipfile`
      - lock file enables D-build by taking a snapshot of all versions of pkgs in an env 
    - now another dev wants to make changes, he would get the code and pipfile and `pipenv install --dev` getting the pytest etc.

- In conflict versions of core depens `Warning: Your dependencies could not be resolved. ...`

  - show depen graph `pipenv graph`  -> tree structure or reverse `--reverse` showing sub-depens with the parent that requires it

- Pipfile - TOML file 

  - ```toml
    [[source]]
    url = "https://pypi.python.org/simple"
    verify_ssl = true
    name = "pypi"
    
    [dev-packages]
    pytest = "*"
    
    [packages]
    flask = "==0.12.1"
    numpy = "*"
    requests = {git = "https://github.com/requests/requests.git", editable = true}
    
    [requires]
    python_version = "3.6"
    ```

  - Ideally should not have any sub-depen in Pipfile and ONLY include pkg imported and used - pipenv will auto-install sub-sub-depens

- Pipfile.lock 

  - JSON with hashes
  - even sub-depens like werkzeug that aren't in Pipfile appear in lock, with hashes used to ensure you're retrieving the same pkg as did in dev

- Extra Feature

  - open 3PPkg in defualt editor `pipenv open flask`
  - run a cmd in venv without shell `pipenv run <cmd>`
  - check security PEP508 `pipenv check`
  - utterly wipe all `pipenv uninstall --all`
  - where venv is `pipenv --venv`
  - where project home is `pipenv --where`



### Package Distribution with Pipenv

- First, `setup.py` is necessary when using `setuptools` as build/dist system - de facto but changing
- For `setup.py` way
  - `install_requires` keyword should include whatever the package "minimally needs to run right"
  - Pipfile
  - Represents the concrete requirements for package
  - Pull the MRD from `setup.py` by installing your package using pipenv:
    - `pipenv install -e .`
    - resulting in a line in Pipfile `e18339a8 = {path = ".", editable = true}`
  - Pipfile.lock
  - Details for a reproducible env generated from pipenv lock



## Flask Project Repo

```shell
flaskr/
│
├── flaskr/
│   ├── ___init__.py
│   ├── db.py
│   ├── schema.sql
│   ├── auth.py
│   ├── blog.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── auth/
│   │   │   ├── login.html
│   │   │   └── register.html
│   │   │
│   │   └── blog/
│   │       ├── create.html
│   │       ├── index.html
│   │       └── update.html
│   │ 
│   └── static/
│       └── style.css
│
├── tests/
│   ├── conftest.py
│   ├── data.sql
│   ├── test_factory.py
│   ├── test_db.py
│   ├── test_auth.py
│   └── test_blog.py
│
├── venv/
│
├── .gitignore
├── setup.py
└── MANIFEST.in
```


[![Build Status](https://github.com/ladybug-tools/honeybee-radiance-postprocess/workflows/CI/badge.svg)](https://github.com/ladybug-tools/honeybee-radiance-postprocess/actions)
[![Coverage Status](https://coveralls.io/repos/github/ladybug-tools/honeybee-radiance-postprocess/badge.svg?branch=master)](https://coveralls.io/github/ladybug-tools/honeybee-radiance-postprocess)

[![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)](https://www.python.org/downloads/release/python-270/) [![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

# honeybee-radiance-postprocess

Postprocessing of Radiance results and matrices

## Installation
```console
pip install honeybee-radiance-postprocess
```

## QuickStart
```python
import honeybee_radiance_postprocess

```

## [API Documentation](http://ladybug-tools.github.io/honeybee-radiance-postprocess/docs)

## Local Development
1. Clone this repo locally
```console
git clone git@github.com:ladybug-tools/honeybee-radiance-postprocess

# or

git clone https://github.com/ladybug-tools/honeybee-radiance-postprocess
```
2. Install dependencies:
```console
cd honeybee-radiance-postprocess
pip install -r dev-requirements.txt
pip install -r requirements.txt
```

3. Run Tests:
```console
python -m pytest tests/
```

4. Generate Documentation:
```console
sphinx-apidoc -f -e -d 4 -o ./docs ./honeybee_radiance_postprocess
sphinx-build -b html ./docs ./docs/_build/docs
```

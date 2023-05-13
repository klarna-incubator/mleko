# ML-Ekosystem

[![PyPI](https://img.shields.io/pypi/v/mleko.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/mleko.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/mleko)][python version]
[![License](https://img.shields.io/pypi/l/mleko)][license]

[![Read the documentation at https://mleko.readthedocs.io/](https://img.shields.io/readthedocs/mleko/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/ErikBavenstrand/mleko/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/ErikBavenstrand/mleko/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/mleko/
[status]: https://pypi.org/project/mleko/
[python version]: https://pypi.org/project/mleko
[read the docs]: https://mleko.readthedocs.io/
[tests]: https://github.com/ErikBavenstrand/mleko/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/ErikBavenstrand/mleko
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

MLEKO is designed to streamline the model building process with its comprehensive set of features:

- Data processing support for different data sources and formats, as well as feature engineering.
- Customizable data processing pipelines with pre-built pipeline steps for various tasks.
- Efficient caching of method call results using caching mixins and fingerprinting utilities.
- Utility functions for logging, decorating, file management, and TQDM wrappers.

## Requirements

- Python 3.8, 3.9, or 3.10
- boto3 >= 1.26.91
- botocore >= 1.29.91
- tqdm >= 4.65.0
- vaex >= 4.16.0

## Installation

You can install _ML-Ekosystem_ via [pip] from [PyPI]:

```console
$ pip install mleko
```

## Usage

- TODO

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_ML-Ekosystem_ is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

<!-- github-only -->

[pypi]: https://pypi.org/
[pip]: https://pip.pypa.io/
[file an issue]: https://github.com/ErikBavenstrand/mleko/issues
[license]: https://github.com/ErikBavenstrand/mleko/blob/main/LICENSE
[contributor guide]: https://github.com/ErikBavenstrand/mleko/blob/main/CONTRIBUTING.md

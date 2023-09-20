# ML-Ekosystem

[![Latest](https://img.shields.io/pypi/v/mleko.svg)][pypi]
[![Status](https://img.shields.io/pypi/status/mleko.svg)][status]
[![License](https://img.shields.io/pypi/l/mleko)][license]
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mleko)][pypi downloads]

[![Read the documentation at https://mleko.readthedocs.io/](https://img.shields.io/readthedocs/mleko/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/ErikBavenstrand/mleko/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/ErikBavenstrand/mleko/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi]: https://pypi.org/project/mleko/
[status]: https://pypi.org/project/mleko/
[pypi downloads]: https://pypi.org/project/mleko/
[read the docs]: https://mleko.readthedocs.io/
[tests]: https://github.com/ErikBavenstrand/mleko/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/ErikBavenstrand/mleko
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

## Features

ML-Ekosystem (`mleko`) is a comprehensive Python library designed to facilitate and streamline the entire model building process. It's built with modularity in mind, encompassing a range of functionalities divided into distinct submodules.

Key features include:

- **Ingest:** Fetches data from a variety of sources, including `AWS S3` and `Kaggle`, ensuring seamless data integration.

- **Convert:** Transforms data between different file formats, with a particular emphasis on `CSV` to `Vaex DataFrame` conversions.
- **Split:** Divide DataFrames into multiple subsets, using a variety of different splitting methods.
- **Feature Selection:** A suite of feature selection methods and algorithms.
- **Transformation:** Includes a variety of data transformation methods, such as `Frequency Encoding` and `Standardization`.
- **Model:** Core model functionalities, allowing for the creation and tuning of a wide array of machine learning models.
- **Pipeline:** Combines all the above functionalities into a single, easy-to-use directed acyclic graph (DAG) pipeline suitable for reproducible model building.

With `mleko`, you have all the tools you need to build robust and efficient machine learning models, all in one place.

## Installation

You can install `mleko` via [pip] from [PyPI]:

```console
$ pip install mleko
```

## Usage

See the [documentation][read the docs] for more information or check out the usage [examples](https://github.com/ErikBavenstrand/mleko/tree/main/examples).

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
`mleko`is free and open source software.

## Issues

If you encounter any problems,
please [file an issue] along with a detailed description.

<!-- github-only -->

[pypi]: https://pypi.org/
[pip]: https://pip.pypa.io/
[file an issue]: https://github.com/ErikBavenstrand/mleko/issues
[license]: https://github.com/ErikBavenstrand/mleko/blob/main/LICENSE
[contributor guide]: https://github.com/ErikBavenstrand/mleko/blob/main/CONTRIBUTING.md

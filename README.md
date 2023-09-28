[![Latest](https://img.shields.io/pypi/v/mleko.svg)][pypi]
[![Status](https://img.shields.io/pypi/status/mleko.svg)][status]
[![License](https://img.shields.io/pypi/l/mleko)][license]
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mleko)][pypi downloads]

[![Read the documentation at https://mleko.readthedocs.io/](https://img.shields.io/readthedocs/mleko/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/ErikBavenstrand/mleko/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/ErikBavenstrand/mleko/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

# `mleko`: Streamlining Machine Learning Pipelines in Python

Simplify and accelerate your machine learning development with `mleko`. Designed with modularity and customization in mind, it seamlessly integrates into your existing workflows. Its robust caching system optimizes performance, taking you from data ingestion to finalized models with unparalleled efficiency.

## Features

`mleko` is engineered to address the end-to-end needs of machine learning pipelines, providing robust, scalable solutions for data science challenges:

- Ingest: Seamlessly integrates with data sources like AWS S3 and Kaggle, offering hassle-free data ingestion and compatibility.
- Convert: Specializes in data format transformations, prominently featuring high-performance conversions from `CSV` to `Vaex DataFrame`, to make your data pipeline-ready.
- Split: Employs sophisticated data partitioning algorithms, allowing you to segment DataFrames into train, test, and validation sets for effective model training and evaluation.
- Feature Selection: Equipped with a suite of feature selection techniques, `mleko` enables model performance by focusing on the most impactful variables.
- Transformation: Facilitates data manipulations such as Frequency Encoding and Standardization, ensuring that your data conforms to the prerequisites of the machine learning algorithms.
- Model: Provides a core set of functionalities for machine learning models, including in-built support for hyperparameter tuning, thereby streamlining the path from data to deployable model.
- Pipeline: Unifies the entire workflow into an intuitive directed acyclic graph (`DAG`) architecture, promoting reproducibility and reducing iteration time and time-to-market for machine learning models.

By integrating these features, `mleko` serves as a comprehensive toolkit for machine learning practitioners looking to build robust models efficiently.

## Installation

You can install `mleko` via `pip` from [PyPI]:

```console
$ pip install mleko
```

# Usage & Examples

See the [documentation][read the docs] for more information or check out the usage [examples](https://github.com/ErikBavenstrand/mleko/tree/main/examples) on well-known datasets like the [Titanic Dataset](https://github.com/ErikBavenstrand/mleko/tree/main/examples/Titanic.ipynb).

## Contributing

We are open to, and grateful for, any contributions made by the community.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license], `mleko` is free and open source software.

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

<!-- github-only -->

[pypi]: hhttps://pypi.org/project/mleko/
[file an issue]: https://github.com/ErikBavenstrand/mleko/issues
[license]: https://github.com/ErikBavenstrand/mleko/blob/main/LICENSE
[contributor guide]: https://github.com/ErikBavenstrand/mleko/blob/main/CONTRIBUTING.md
[status]: https://pypi.org/project/mleko/
[pypi downloads]: https://pypi.org/project/mleko/
[read the docs]: https://mleko.readthedocs.io/
[tests]: https://github.com/ErikBavenstrand/mleko/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/ErikBavenstrand/mleko
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

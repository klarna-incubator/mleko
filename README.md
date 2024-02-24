# mleko: Streamlining Machine Learning Pipelines in Python

Simplify and accelerate your machine learning development with `mleko`. Designed with modularity and customization in mind, it seamlessly integrates into your existing workflows. Its robust caching system optimizes performance, taking you from data ingestion to finalized models with unparalleled efficiency.

[![Developed at Klarna][klarna-image]][klarna-url]
[![License](https://img.shields.io/pypi/l/mleko)][licence]
[![Static Badge](https://img.shields.io/badge/docs-pages-blue)][pages]

[![Latest](https://img.shields.io/pypi/v/mleko.svg)][pypi]
[![Status](https://img.shields.io/pypi/status/mleko.svg)][pypi]
[![PyPI - Downloads](https://img.shields.io/pypi/dm/mleko)][pypi]
[![Python Version](https://img.shields.io/pypi/pyversions/mleko)][pypi]

[![Tests](https://github.com/klarna-incubator/mleko/workflows/Tests/badge.svg)][tests]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

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

## Usage & Examples

See the [documentation][pages] for more information or check out the usage [examples](https://github.com/klarna-incubator/mleko/tree/main/examples) on well-known datasets like the [Titanic Dataset](https://github.com/klarna-incubator/mleko/tree/main/examples/Titanic.ipynb).

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

## Contributing

We are open to, and grateful for, any contributions made by the community.
To learn more, see the [Contributor Guide].

## Release History

See our [changelog](https://github.com/klarna-incubator/mleko/tree/main/CHANGELOG.md).

## Acknowledgements

The development of `mleko` was significantly influenced by existing work of the following individuals:

- **Felipe Breve Siola** ([fsiola](https://github.com/fsiola))
- **Sai Ma** ([metanouvelle](https://github.com/metanouvelle))
- **Ahmet Anil Pala** ([aanilpala](https://github.com/aanilpala))

Their insights and contributions provided a solid foundation for this library. We appreciate their effort and recognize their contributions that led to the creation of `mleko`.

## License

Copyright © 2024 Klarna Bank AB

For license details, see the [LICENSE](https://github.com/klarna-incubator/mleko/blob/main/LICENSE) file in the root of this project.

[klarna-image]: https://img.shields.io/badge/%20-Developed%20at%20Klarna-black?style=round-square&labelColor=ffb3c7&logo=klarna&logoColor=black
[klarna-url]: https://klarna.github.io
[pypi]: https://pypi.org/project/mleko/
[licence]: https://github.com/klarna-incubator/mleko/blob/main/LICENSE
[tests]: https://github.com/klarna-incubator/mleko/actions?workflow=Tests
[pages]: https://klarna-incubator.github.io/mleko/
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black
[contributor guide]: https://github.com/klarna-incubator/mleko/blob/main/CONTRIBUTING.md
[file an issue]: https://github.com/klarna-incubator/mleko/issues

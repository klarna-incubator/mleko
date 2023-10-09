# Contributor Guide

Thank you for your interest in improving this `mleko`. Please read the following to better understand how to ask questions or work on something.
This project is open-source under the [Apache-2.0 license] and
welcomes contributions in the form of bug reports, feature requests, and pull requests.

All members of our community are expected to follow our [Code of Conduct]. Please make sure you are welcoming and friendly in all of our spaces.

Here is a list of important resources for contributors:

- [Source Code]
- [Documentation]
- [Issue Tracker]

[apache-2.0 license]: https://www.apache.org/licenses/LICENSE-2.0
[source code]: https://github.com/klarna-incubator/mleko
[documentation]: https://mleko.readthedocs.io/
[issue tracker]: https://github.com/klarna-incubator/mleko/issues

## How to report a bug

Report bugs on the [Issue Tracker].

When filing an issue, make sure to answer these questions:

- Which operating system and Python version are you using?
- Which version of `mleko` are you using?
- What did you do?
- What did you expect to see?
- What did you see instead?

The best way to get your bug fixed is to provide a test case,
and/or steps to reproduce the issue.

## How to request a feature

Request features on the [Issue Tracker].

## How to set up your development environment

You need Python 3.8+ and the following tools:

- [poetry](https://python-poetry.org/)
- [nox](https://nox.thea.codes/)
- [nox-poetry](https://nox-poetry.readthedocs.io/)

Install the package with development requirements:

```console
$ poetry install
```

[poetry]: https://python-poetry.org/
[nox]: https://nox.thea.codes/
[nox-poetry]: https://nox-poetry.readthedocs.io/

## How to test the project

Run the full test suite:

```console
$ nox
```

List the available Nox sessions:

```console
$ nox --list-sessions
```

You can also run a specific Nox session.
For example, invoke the unit test suite like this:

```console
$ nox --session=tests
```

Unit tests are located in the _tests_ directory,
and are written using the [pytest] testing framework.

[pytest]: https://pytest.readthedocs.io/

## How to submit changes

At Klarna, we strive toward achieving the highest possible quality for our products. Therefore, we require you to follow these guidelines if you wish to open a [pull request] to submit changes to this project.

Your contribution has to meet the following criteria:

- It is accompanied by a description regarding what has been changed and why.
- Pull requests should implement a boxed change, meaning they should optimally not try to address many things at once.
- All code and documentation must follow the style specified by the included configuration.
- The Nox test suite must pass without errors.
- Include unit tests. This project maintains 100% code coverage.
- If your changes add functionality, update the documentation accordingly.

Feel free to submit early, though—we can always iterate on this.

To run linting and code formatting checks before committing your change, you can install pre-commit as a Git hook by running the following command:

```console
$ nox --session=pre-commit -- install
```

It is recommended to open an issue before starting work on anything.
This will allow a chance to talk it over with the owners and validate your approach.

[pull request]: https://github.com/klarna-incubator/mleko/pulls
[code of conduct]: https://github.com/klarna-incubator/.github/blob/main/CODE_OF_CONDUCT.md

"""`mleko`: Streamlining Machine Learning Pipelines in Python.

Simplify and accelerate your machine learning development with `mleko`. Designed with modularity and
customization in mind, it seamlessly integrates into your existing workflows. Its robust caching system
optimizes performance, taking you from data ingestion to finalized models with unparalleled efficiency.

`mleko` is engineered to address the end-to-end needs of machine learning pipelines, providing robust,
scalable solutions for data science challenges:

* Ingest: Seamlessly integrates with data sources like AWS S3 and Kaggle, offering hassle-free data ingestion and
  compatibility.

* Convert: Specializes in data format transformations, prominently featuring high-performance conversions from CSV to
  Vaex DataFrame, to make your data pipeline-ready.

* Split: Employs sophisticated data partitioning algorithms, allowing you to segment DataFrames into train, test, and
  validation sets for effective model training and evaluation.

* Feature Selection: Equipped with a suite of feature selection techniques, `mleko` enables model performance by
  focusing on the most impactful variables.

* Transformation: Facilitates data manipulations such as Frequency Encoding and Standardization, ensuring that your
  data conforms to the prerequisites of the machine learning algorithms.

* Model: Provides a core set of functionalities for machine learning models, including in-built support for
  hyperparameter tuning, thereby streamlining the path from data to deployable model.

* Pipeline: Unifies the entire workflow into an intuitive directed acyclic graph (DAG) architecture, promoting
  reproducibility and reducing iteration time and time-to-market for machine learning models.

By integrating these features, `mleko` serves as a comprehensive toolkit for machine learning
practitioners looking to build robust models efficiently.
"""

from __future__ import annotations


__version__ = "2.2.0"

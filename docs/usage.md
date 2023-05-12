# Usage

## Overview

This guide demonstrates how to use the mleko library to fetch and manage dataset files from Kaggle and convert them to Apache Arrow format. In this example, we are processing the "mlg-ulb/creditcardfraud" dataset.

## Credit Card Fraud Example

Import the required classes:

```python
from mleko.data.sources import KaggleDataSource
from mleko.data.converters import CsvToArrowConverter
from mleko.pipeline import Pipeline
from mleko.pipeline.steps import IngestStep, ConvertStep
```

Set the dataset name and initialize KaggleDataSource and CsvToArrowConverter objects:

```python
DATASET_NAME = "mlg-ulb/creditcardfraud"
kaggle_data_source = KaggleDataSource(f"data/{DATASET_NAME}/raw", owner_slug=DATASET_NAME.split("/")[0], dataset_slug=DATASET_NAME.split("/")[1])
csv_to_arrow_converter = CsvToArrowConverter(output_directory=f"data/{DATASET_NAME}/converted", downcast_float=True)
```

Create a Pipeline instance with IngestStep and ConvertStep:

```python
pipeline = Pipeline(steps=[
    IngestStep(kaggle_data_source),
    ConvertStep(csv_to_arrow_converter)
])
```

Run the pipeline and display the first few records of the resulting DataFrame:

```python
df = pipeline.run().data
df.head()
```

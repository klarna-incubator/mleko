"""This module contains helper functions for working with `vaex` DataFrames."""

from __future__ import annotations

import vaex


def get_column(df: vaex.DataFrame, column: str) -> vaex.Expression:
    """Get specified column from a DataFrame as an Expression.

    Args:
        df: The input DataFrame.
        column: The name of the desired column.

    Returns:
        The specified column as an Expression.
    """
    return df[column]


def get_columns(df: vaex.DataFrame, columns: list[str]) -> vaex.DataFrame:
    """Get specified columns from a DataFrame.

    Args:
        df: The input DataFrame.
        columns: A list of the names of the desired columns or an Expression.

    Returns:
        A DataFrame containing only the specified columns.

    Examples:
        >>> import vaex
        >>> from mleko.utils import get_columns
        >>> df = vaex.from_arrays(column1=[1, 2, 3], column2=[4, 5, 6], column3=[7, 8, 9])
        >>> get_columns(df, ["column1", "column3"]).get_column_names()
        ['column1', 'column3']
    """
    return df[columns]


def get_filtered_df(df: vaex.DataFrame, filter: vaex.Expression) -> vaex.DataFrame:
    """Get filtered DataFrame.

    Will return a DataFrame containing only the rows that satisfy the filter. The filter is an Expression that
    evaluates to a boolean value for each row.

    Args:
        df: The input DataFrame.
        filter: A boolean Expression used to filter the DataFrame.

    Returns:
        A DataFrame containing only the rows that satisfy the filter.

    Examples:
        >>> import vaex
        >>> from mleko.utils import get_filtered_df
        >>> df = vaex.from_arrays(column1=[1, 2, 3], column2=[4, 5, 6], column3=[7, 8, 9])
        >>> get_filtered_df(df, df.column1 > 1)
        #    column1    column2    column3
        0          2          5          8
        1          3          6          9
    """
    return df[filter]


def get_indices(df: vaex.DataFrame, indices: list[int]) -> vaex.DataFrame:
    """Get DataFrame containing only the specified indices.

    Args:
        df: The input DataFrame.
        indices: A list of the indices to be extracted.

    Returns:
        A DataFrame containing only the specified indices.

    Examples:
        >>> import vaex
        >>> from mleko.utils import get_indices
        >>> df = vaex.from_arrays(column1=[1, 2, 3], column2=[4, 5, 6], column3=[7, 8, 9])
        >>> get_indices(df, [0, 2])
        #    column1    column2    column3
        0          1          4          7
        1          3          6          9
    """
    idx_name = "index"
    df[idx_name] = vaex.vrange(0, df.shape[0])
    index = get_column(df, idx_name)
    selection = get_filtered_df(df, index.isin(indices))
    selection.delete_virtual_column(idx_name)
    return selection.extract()

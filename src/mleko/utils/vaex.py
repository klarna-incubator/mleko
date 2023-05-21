"""Utility functions for working with Vaex DataFrames.

This module provides utility functions that help with extracting specific columns
and rows from Vaex DataFrames.
"""
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


def get_columns(df: vaex.DataFrame, columns: list[str] | vaex.Expression) -> vaex.DataFrame:
    """Get specified columns from a DataFrame.

    Args:
        df: The input DataFrame.
        columns: A list of the names of the desired columns or an Expression.
            If an Expression is provided, it is assumed to be a boolean mask
            that can be used to filter the DataFrame.

    Returns:
        A DataFrame containing only the specified columns.
    """
    return df[columns]


def get_indices(df: vaex.DataFrame, indices: list[int]) -> vaex.DataFrame:
    """Get specified row indices from a DataFrame.

    Args:
        df: The input DataFrame.
        indices: A list of row indices to extract from the DataFrame.

    Returns:
        A DataFrame containing rows for the specified indices.
    """
    idx_name = "index"
    df[idx_name] = vaex.vrange(0, df.shape[0])
    index = get_column(df, idx_name)
    selection = get_columns(df, index.isin(indices))
    selection.delete_virtual_column(idx_name)
    return selection.extract()

import json
import re
from datetime import date

import pandas as pd
import pandas.testing as pdt
import polars as pl
import pytest
from pyspark.sql.functions import abs as spark_abs
from pyspark.sql.functions import asc as spark_asc
from pyspark.sql.functions import asc_nulls_first as spark_asc_nulls_first
from pyspark.sql.functions import asc_nulls_last as spark_asc_nulls_last
from pyspark.sql.functions import coalesce as spark_coalesce
from pyspark.sql.functions import col as spark_col
from pyspark.sql.functions import concat as spark_concat
from pyspark.sql.functions import dense_rank as spark_dense_rank
from pyspark.sql.functions import desc as spark_desc
from pyspark.sql.functions import desc_nulls_first as spark_desc_nulls_first
from pyspark.sql.functions import desc_nulls_last as spark_desc_nulls_last
from pyspark.sql.functions import expr as spark_expr
from pyspark.sql.functions import get_json_object as spark_get_json_object
from pyspark.sql.functions import length as spark_length
from pyspark.sql.functions import lit as spark_lit
from pyspark.sql.functions import lower as spark_lower
from pyspark.sql.functions import rank as spark_rank
from pyspark.sql.functions import regexp_replace as spark_regexp_replace
from pyspark.sql.functions import round as spark_round
from pyspark.sql.functions import row_number as spark_row_number
from pyspark.sql.functions import struct as spark_struct
from pyspark.sql.functions import to_timestamp as spark_to_timestamp
from pyspark.sql.functions import when as spark_when
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.sql.types import IntegerType as SparkIntegerType
from pyspark.sql.types import MapType, StringType, StructField, StructType
from pyspark.sql.window import Window as SparkWindow

from sparkleframe.polarsdf import Window
from sparkleframe.polarsdf.dataframe import DataFrame
from sparkleframe.polarsdf.functions import (
    abs,
    asc,
    asc_nulls_first,
    asc_nulls_last,
    coalesce,
    col,
    concat,
    dense_rank,
    desc,
    desc_nulls_first,
    desc_nulls_last,
    expr,
    get_json_object,
    length,
    lit,
    lower,
    rank,
    regexp_replace,
    round,
    row_number,
    struct,
    to_timestamp,
    try_element_at,
    try_to_date,
    try_to_timestamp,
    when,
)
from sparkleframe.tests.pyspark_test import assert_pyspark_df_equal
from sparkleframe.tests.utils import create_spark_df, to_records

sample_data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}


def _polars_map_entries_to_spark_dict(obj: object) -> object:
    """Normalize Polars map-as-list-of-{key,value} to dicts for Spark createDataFrame(pdf, schema)."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "key" in v[0] and "value" in v[0]:
                out[k] = {e["key"]: float(e["value"]) for e in v}
            else:
                out[k] = _polars_map_entries_to_spark_dict(v)
        return out
    return obj


@pytest.fixture
def sparkle_df():
    return DataFrame(pl.DataFrame(sample_data))


@pytest.fixture
def spark_df(spark):
    return spark.createDataFrame(pd.DataFrame(sample_data))


class TestFunctions:
    def test_when(self, spark, sparkle_df, spark_df):
        expr = when(col("a") > 2, "yes").otherwise("no")

        # Add the result column to the full Polars DataFrame
        result_spark_df = spark.createDataFrame(sparkle_df.withColumn("result", expr).toPandas())

        # Add result column to full Spark DataFrame
        expected_spark_df = spark_df.withColumn("result", spark_when(spark_col("a") > 2, "yes").otherwise("no"))

        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_expr_sql_on_column(self, spark):
        """sql_expr path: same integral dtype end-to-end (avoid float→long pandas inference)."""
        data = to_records({"x": [1, 2, 3]})
        spark_df = spark.createDataFrame(data).withColumn("y", spark_expr("x * 2"))
        pl_df = DataFrame(pl.DataFrame(data)).withColumn("y", expr("x * 2"))
        result_spark_df = spark.createDataFrame(pl_df.toPandas())
        assert_pyspark_df_equal(result_spark_df, spark_df, ignore_nullable=True)

    def test_expr_uuid(self, spark):
        uuid_pat = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )

        def assert_uuid_strings(sdf, col_name: str) -> None:
            vals = [row[col_name] for row in sdf.collect()]
            assert len(vals) == len(set(vals))
            for v in vals:
                assert isinstance(v, str) and uuid_pat.match(v), v

        data = to_records({"id": [1, 2, 3, 4, 5]})
        spark_df = spark.createDataFrame(data).withColumn("uuid_col", spark_expr("uuid()"))
        pl_df = DataFrame(pl.DataFrame(data)).withColumn("uuid_col", expr("uuid()"))
        assert_uuid_strings(spark_df, "uuid_col")
        assert_uuid_strings(spark.createDataFrame(pl_df.toPandas()), "uuid_col")

    def test_chained_when_boolean_output(self, spark):
        # Input data
        data = to_records({"b": ["A", "B", "C", "D"], "c": ["b", "e", "g", "z"]})

        polars_df = DataFrame(pl.DataFrame(data))
        expr = (
            when((col("b") == "A") & (col("c").isin("A", "b", "c")), True)
            .when((col("b") == "B") & (col("c").isin("d", "e")), True)
            .when((col("b") == "C") & (col("c").isin("f", "g", "h", "i")), True)
            .otherwise(False)
        )

        result_df = polars_df.withColumn("result", expr)
        result_spark_df = spark.createDataFrame(result_df.df.to_dicts())

        # Expected result using PySpark chained when()
        expected_df = spark.createDataFrame(data).withColumn(
            "result",
            spark_when((spark_col("b") == "A") & (spark_col("c").isin("A", "b", "c")), True)
            .when((spark_col("b") == "B") & (spark_col("c").isin("d", "e")), True)
            .when((spark_col("b") == "C") & (spark_col("c").isin("f", "g", "h", "i")), True)
            .otherwise(False),
        )

        # Compare results
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "json_data, path, expected_values",
        [
            ([json.dumps({"a": 1}), json.dumps({"a": 2})], "$.a", ["1", "2"]),
            ([json.dumps({"a": {"b": 3}}), json.dumps({"a": {"b": 4}})], "$.a.b", ["3", "4"]),
            ([json.dumps({"arr": [10, 20]}), json.dumps({"arr": [30, 40]})], "$.arr[1]", ["20", "40"]),
            ([json.dumps({"a": {"b": [5, 6]}}), json.dumps({"a": {"b": [7, 8]}})], "$.a.b[0]", ["5", "7"]),
            (
                [json.dumps({"items": [{"id": 1}, {"id": 2}]}), json.dumps({"items": [{"id": 3}, {"id": 4}]})],
                "$.items[1].id",
                ["2", "4"],
            ),
        ],
    )
    def test_get_json_object(self, spark, json_data, path, expected_values):
        df = pd.DataFrame({"json_col": json_data})

        spark_df = spark.createDataFrame(df)
        expected_df = spark_df.select(spark_get_json_object("json_col", path).alias("result"))

        polars_df = DataFrame(pl.DataFrame(df))
        result_df = polars_df.select(get_json_object("json_col", path).alias("result"))
        result_spark_df = spark.createDataFrame(result_df.toPandas())

        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "literal_value",
        [
            42,  # int
            3.14,  # float
            "hello",  # string
            True,  # boolean
            None,  # null
        ],
    )
    def test_lit_against_spark(self, spark, literal_value):
        df = pl.DataFrame({"x": [1, 2, 3]})
        sparkle_df = DataFrame(df)
        result_df = sparkle_df.select(lit(literal_value).alias("value")).toPandas()

        # Result using Spark
        spark_df = spark.createDataFrame(pd.DataFrame({"x": [1, 2, 3]}))
        expected_df = spark_df.select(spark_lit(literal_value).alias("value")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,  # Important: ignores schema/type mismatches
        )

    @pytest.mark.parametrize(
        "a_vals, b_vals, expected_vals",
        [
            ([None, 2, None], [1, None, 3], [1, 2, 3]),
            ([None, None, None], [None, None, None], [None, None, None]),
            ([None, 5, 6], ["x", "y", None], ["x", 5, 6]),
            (["", None, "z"], ["a", "b", None], ["", "b", "z"]),
        ],
    )
    def test_coalesce_against_spark(self, spark, a_vals, b_vals, expected_vals):
        # Build pandas DataFrame for both Spark and Polars
        data = to_records({"a": a_vals, "b": b_vals})

        # Spark setup
        if expected_vals == [None, None, None]:
            spark_df = spark.createDataFrame(data=[("1", "1")], schema="a: string, b: string")
            spark_df = spark_df.withColumn("a", spark_lit(None)).withColumn("b", spark_lit(None))

            expected_spark_df = spark.createDataFrame(data=[("1", "1")], schema="a: string, b: string")
            expected_spark_df = expected_spark_df.withColumn("a", spark_lit(None)).withColumn("b", spark_lit(None))
        else:
            spark_df = spark.createDataFrame(data)
            expected_spark_df = spark_df.select(spark_coalesce(spark_col("a"), spark_col("b")).alias("result"))

        # sparkleframe setup
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(coalesce(col("a"), col("b")).alias("result"))

        if result_df.df.to_dicts() == [{"result": None}, {"result": None}, {"result": None}]:
            result_spark_df = spark_df.withColumn("a", spark_lit(None)).withColumn("b", spark_lit(None))
        else:
            result_spark_df = spark.createDataFrame(result_df.df.to_dicts())

        # Compare using PySpark equality
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "values, scale",
        [
            ([1.234, 2.345, 3.456], 0),  # round to integer
            ([1.234, 2.345, 3.456], 1),  # round to 1 decimal
            ([1.234, 2.345, 3.456], 2),  # round to 2 decimals
            ([None, 2.555, 3.666], 1),  # include None
        ],
    )
    def test_round_against_spark(self, spark, values, scale):
        data = to_records({"x": values})

        # Sparkleframe / Polars
        polars_df = DataFrame(pl.DataFrame(data))
        result_df = polars_df.select(round(col("x"), scale).alias("rounded")).toPandas()

        # PySpark
        spark_df = spark.createDataFrame(data)
        expected_df = spark_df.select(spark_round(spark_col("x"), scale).alias("rounded")).toPandas()

        # Compare using pandas
        pdt.assert_frame_equal(
            result_df.reset_index(drop=True),
            expected_df.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-5,
        )

    @pytest.mark.parametrize(
        "data, column",
        [
            ({"x": [3, 1, 2]}, "x"),
            ({"x": ["b", "c", "a"]}, "x"),
            ({"x": [3.3, 1.1, 2.2]}, "x"),
        ],
    )
    def test_asc_against_spark(self, spark, data, column):
        data = to_records(data)
        # Create input DataFrame
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(data)

        # SparkleFrame: order by asc
        result_df = polars_df.df.sort(asc(col(column)).to_native())
        result_spark_df = create_spark_df(spark, result_df)

        # PySpark: order by asc
        expected_df = spark_df.orderBy(spark_asc(column))

        # Compare using PySpark equality
        assert_pyspark_df_equal(result_spark_df.orderBy("x"), expected_df.orderBy("x"), ignore_nullable=True)

    @pytest.mark.parametrize(
        "data, column",
        [
            ({"x": [3, 1, 2]}, "x"),
            ({"x": ["b", "c", "a"]}, "x"),
            ({"x": [3.3, 1.1, 2.2]}, "x"),
        ],
    )
    def test_desc_against_spark(self, spark, data, column):
        data = to_records(data)
        # Create input DataFrame
        polars_df = DataFrame(pl.DataFrame(data))
        spark_df = spark.createDataFrame(data)

        # SparkleFrame: order by desc
        result_df = polars_df.df.sort(desc(col(column)).to_native())
        result_spark_df = create_spark_df(spark, result_df)

        # PySpark: order by desc
        expected_df = spark_df.orderBy(spark_desc(column))

        # Compare using PySpark equality
        assert_pyspark_df_equal(result_spark_df.orderBy("x"), expected_df.orderBy("x"), ignore_nullable=True)

    @pytest.mark.parametrize(
        "col_input",
        [
            "txt",
            col("txt"),
        ],
    )
    @pytest.mark.parametrize(
        "input_values, pattern, replacement, expected_values",
        [
            (["abc123", "xyz456"], r"\d+", "", ["abc", "xyz"]),  # Remove digits
            (["hello world", "world hello"], "world", "earth", ["hello earth", "earth hello"]),  # Replace word
            (["aaa", "aba", "aca"], "a", "x", ["xxx", "xbx", "xcx"]),  # Replace all a's
            (["test123", "123test"], r"^\d+", "NUM", ["test123", "NUMtest"]),  # Match digits at start
            (["test123", "123test"], r"\d+$", "END", ["testEND", "123test"]),  # Match digits at end
        ],
    )
    def test_regexp_replace_str_vs_column(self, spark, col_input, pattern, replacement, input_values, expected_values):
        # Prepare input as pandas DataFrame and convert to Spark
        df_data = pd.DataFrame({"txt": input_values})
        spark_input_df = spark.createDataFrame(df_data)

        # Expected Spark result
        expected_df = spark_input_df.select(spark_regexp_replace("txt", pattern, replacement).alias("replaced"))

        # SparkleFrame Polars-based result
        polars_df = DataFrame(pl.DataFrame(df_data))
        result_df = polars_df.select(regexp_replace(col_input, pattern, replacement).alias("replaced"))
        result_spark_df = spark.createDataFrame(result_df.df.to_dicts())

        # Validate against PySpark
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "input_values",
        [
            ["abc", "de", ""],  # basic strings
            ["你好", "世界", ""],  # unicode characters
            [None, "x", "longer string"],  # includes None
            ["😊", "👍🏽", "💯"],  # emojis with multiple bytes
        ],
    )
    @pytest.mark.parametrize("col_input", ["txt", col("txt")])
    def test_length_str_vs_column(self, spark, input_values, col_input):
        # Prepare pandas and polars DataFrame
        df_data = pd.DataFrame({"txt": input_values})
        polars_df = DataFrame(pl.DataFrame(df_data))

        # PySpark DataFrame for expected result
        spark_df = spark.createDataFrame(df_data)
        expected_df = spark_df.select(spark_length("txt").alias("result"))

        # SparkleFrame/Polars result
        result_df = polars_df.select(length(col_input).alias("result"))
        result_spark_df = spark.createDataFrame(result_df.df.to_dicts()).withColumn(
            "result", spark_col("result").cast(SparkIntegerType())
        )

        # Assert equality
        assert_pyspark_df_equal(result_spark_df, expected_df)

    @pytest.mark.parametrize(
        "col_input",
        [
            "ts",
            col("ts"),
        ],
    )
    @pytest.mark.parametrize(
        "datetime_strs, fmt",
        [
            # Standard format
            (["2023-01-01 12:34:56", "2024-02-02 23:45:01"], "yyyy-MM-dd HH:mm:ss"),
            # Day-first format
            (["01-03-2023 09:15:00", "31-12-2022 23:59:59"], "dd-MM-yyyy HH:mm:ss"),
            # Compact format
            (["20230101 120000", "20240101 130000"], "yyyyMMdd HHmmss"),
            # Millisecond precision (3 digits)
            (["2024-05-31 20:14:19.993", "2023-12-12 11:11:11.123"], "yyyy-MM-dd HH:mm:ss.SSS"),
            # Microsecond precision (6 digits)
            (["2024-05-31 23:58:32.880000", "2023-12-12 11:11:11.123456"], "yyyy-MM-dd HH:mm:ss.SSSSSS"),
            # Single-digit millisecond
            (["2024-05-31 20:14:19.9", "2023-12-12 11:11:11.1"], "yyyy-MM-dd HH:mm:ss.S"),
            # Two-digit millisecond
            (["2024-05-31 20:14:19.99", "2023-12-12 11:11:11.12"], "yyyy-MM-dd HH:mm:ss.SS"),
            # Five-digit fractional seconds (partial microseconds)
            (["2024-05-31 20:14:19.12345", "2023-12-12 11:11:11.99999"], "yyyy-MM-dd HH:mm:ss.SSSSS"),
        ],
    )
    def test_to_timestamp_against_spark(self, spark, col_input, datetime_strs, fmt):
        # Create input DataFrame
        df = pd.DataFrame({"ts": datetime_strs})
        polars_df = DataFrame(pl.DataFrame(df))

        # Spark expected output
        spark_df = spark.createDataFrame(df)
        expected_df = spark_df.select(spark_to_timestamp("ts", fmt).alias("result"))

        # Sparkleframe / Polars output
        result_df = create_spark_df(spark, polars_df.select(to_timestamp(col_input, fmt).alias("result")))

        assert_pyspark_df_equal(result_df, expected_df)

    @pytest.mark.parametrize(
        "sparkle_col_type, spark_col_type",
        [
            (str, str),
            (col, spark_col),
        ],
    )
    @pytest.mark.parametrize(
        "sparkle_func, spark_func",
        [
            (rank, spark_rank),
            (dense_rank, spark_dense_rank),
            (row_number, spark_row_number),
        ],
    )
    @pytest.mark.parametrize(
        "partition_cols, order_cols",
        [
            (["group"], [("category", "asc")]),
            (["group"], [("category", "asc_nulls_first")]),
            (["group"], [("category", "asc_nulls_last")]),
            (["group"], [("value", "desc")]),
            (["group", "subcategory"], [("value", "asc")]),
            (["group", "subcategory"], [("value", "desc"), ("category", "asc")]),
            (["group", "category"], [("subcategory", "asc"), ("value", "desc")]),
        ],
    )
    def test_ranks_with_subcategory(
        self, spark, sparkle_col_type, spark_col_type, sparkle_func, spark_func, partition_cols, order_cols
    ):
        # Extended sample data with `subcategory`
        # Expanded dataset for more exhaustive testing
        data = [
            {"group": "A", "category": "A", "subcategory": "alpha", "value": 100},
            {"group": "A", "category": "A", "subcategory": "alpha", "value": 100},
            {"group": "A", "category": "A", "subcategory": "alpha", "value": 200},
            {"group": "A", "category": "A", "subcategory": "beta", "value": 50},
            {"group": "A", "category": "B", "subcategory": "alpha", "value": 120},
            {"group": "A", "category": "B", "subcategory": "beta", "value": 180},
            {"group": "B", "category": "A", "subcategory": "alpha", "value": 300},
            {"group": "B", "category": "A", "subcategory": "beta", "value": 310},
            {"group": "B", "category": "B", "subcategory": "alpha", "value": 150},
            {"group": "B", "category": "B", "subcategory": "beta", "value": 160},
            {"group": "B", "category": "B", "subcategory": "beta", "value": 170},
            {"group": "C", "category": "A", "subcategory": "alpha", "value": 80},
            {"group": "C", "category": "A", "subcategory": "alpha", "value": 85},
            {"group": "C", "category": "A", "subcategory": "beta", "value": 70},
            {"group": "C", "category": "B", "subcategory": "beta", "value": 60},
            {"group": "C", "category": "B", "subcategory": "beta", "value": 100},
        ]

        order_func = {
            "asc": (asc, spark_asc),
            "asc_nulls_first": (asc_nulls_first, spark_asc_nulls_first),
            "asc_nulls_last": (asc_nulls_last, spark_asc_nulls_last),
            "desc": (desc, spark_desc),
            "desc_nulls_first": (desc_nulls_first, spark_desc_nulls_first),
            "desc_nulls_last": (desc_nulls_last, spark_desc_nulls_last),
        }

        pl_df = pl.DataFrame(data)

        # Build Sparkle DataFrame
        sparkle_df = DataFrame(pl_df)

        # Convert to order expressions
        order_exprs = [order_func[direction][0](col) for col, direction in order_cols]

        # Apply rank over window
        sparkle_df = sparkle_df.withColumn(
            "rank",
            sparkle_func().over(
                Window.partitionBy(*[sparkle_col_type(col) for col in partition_cols]).orderBy(*order_exprs)
            ),
        )

        # Cast and sort for stable comparison
        sparkle_df = (
            create_spark_df(spark, sparkle_df)
            .withColumn("rank", spark_col("rank").cast(SparkIntegerType()))
            .orderBy("group", "category", "subcategory", "value")
        )

        # Build Spark reference DataFrame
        spark_df = create_spark_df(spark, pl_df)

        spark_order_exprs = [order_func[direction][1](col) for col, direction in order_cols]

        spark_df = spark_df.withColumn(
            "rank",
            spark_func().over(
                SparkWindow.partitionBy(*[spark_col_type(col) for col in partition_cols]).orderBy(*spark_order_exprs)
            ),
        ).orderBy("group", "category", "subcategory", "value")

        assert_pyspark_df_equal(sparkle_df, spark_df, ignore_nullable=True)

    @pytest.mark.parametrize(
        "values",
        [
            [-5, -1, 0, 1, 5],  # integers
            [-3.5, -0.1, 0.0, 0.1, 3.5],  # floats
            [None, -2, 2, None],  # include None
        ],
    )
    def test_abs_against_spark(self, spark, values):
        data = to_records({"x": values})

        # Sparkleframe Polars
        sf_df = DataFrame(pd.DataFrame(data))
        result_sf = sf_df.select(abs(col("x")).alias("abs_x"))
        result_spark_df = create_spark_df(spark, result_sf)

        # PySpark
        spark_df = spark.createDataFrame(pd.DataFrame(data))
        expected_df = spark_df.select(spark_abs("x").alias("abs_x"))

        # Compare
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True, allow_nan_equality=True)

    @pytest.mark.parametrize("col_input", ["txt", col("txt")])
    @pytest.mark.parametrize(
        "input_values",
        [
            ["ABC", "abc", "AbC"],  # mixed case
            ["", None, "Already lower"],  # empty and None
            ["MiXeD 123!@$", "CamelCase", "UPPER lower"],  # with numbers/symbols
        ],
    )
    def test_lower_str_vs_column(self, spark, col_input, input_values):
        # Prepare input data
        df_data = pd.DataFrame({"txt": input_values})

        # Expected Spark result
        spark_input_df = spark.createDataFrame(df_data)
        expected_df = spark_input_df.select(spark_lower("txt").alias("lowered"))

        # SparkleFrame / Polars result
        polars_df = DataFrame(pl.DataFrame(df_data))
        result_sf = polars_df.select(lower(col_input).alias("lowered"))
        result_spark_df = create_spark_df(spark, result_sf)

        # Compare
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)

    def test_struct_oracle(self, spark, sparkle_df, spark_df):
        """PySpark parity: field names follow CreateStruct (plain cols vs colN for lit / expr)."""
        exprs = [
            struct("a", "b").alias("s1"),
            struct(col("a"), col("b")).alias("s2"),
            struct([col("a"), col("b")]).alias("s3"),
            struct(col("a"), lit(1)).alias("s4"),
            struct(lit(1), lit(2)).alias("s5"),
            struct(col("a") + 1, col("b")).alias("s6"),
            struct(col("a"), struct(col("b"), lit(3))).alias("s7"),
            struct(col("a").alias("z")).alias("s8"),
        ]
        expected_spark_df = spark_df.select(
            spark_struct("a", "b").alias("s1"),
            spark_struct(spark_col("a"), spark_col("b")).alias("s2"),
            spark_struct([spark_col("a"), spark_col("b")]).alias("s3"),
            spark_struct(spark_col("a"), spark_lit(1)).alias("s4"),
            spark_struct(spark_lit(1), spark_lit(2)).alias("s5"),
            spark_struct(spark_col("a") + 1, spark_col("b")).alias("s6"),
            spark_struct(spark_col("a"), spark_struct(spark_col("b"), spark_lit(3))).alias("s7"),
            spark_struct(spark_col("a").alias("z")).alias("s8"),
        )
        pdf = sparkle_df.select(*exprs).toPandas()
        result_spark_df = spark.createDataFrame(pdf, schema=expected_spark_df.schema)
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_struct_nested_with_array_and_map(self, spark):
        """PySpark parity: struct with array and map fields, plus nested struct."""
        schema = StructType(
            [
                StructField("id", SparkIntegerType(), True),
                StructField("nums", ArrayType(SparkIntegerType()), True),
                StructField("kv", MapType(StringType(), DoubleType()), True),
            ]
        )
        spark_row = (1, [10, 20, 30], {"x": 1.0, "y": 2.0})
        spark_input = spark.createDataFrame([spark_row], schema)
        pl_df = pl.DataFrame(
            {
                "id": [1],
                "nums": [[10, 20, 30]],
                "kv": [[{"key": "x", "value": 1.0}, {"key": "y", "value": 2.0}]],
            }
        )
        sparkle_df = DataFrame(pl_df)
        exprs = [
            struct(col("id"), col("nums"), col("kv")).alias("s1"),
            struct(col("id"), struct(col("nums"), col("kv"))).alias("s2"),
        ]
        expected_spark_df = spark_input.select(
            spark_struct(spark_col("id"), spark_col("nums"), spark_col("kv")).alias("s1"),
            spark_struct(spark_col("id"), spark_struct(spark_col("nums"), spark_col("kv"))).alias("s2"),
        )
        pdf = sparkle_df.select(*exprs).toPandas()
        for col_name in ("s1", "s2"):
            pdf[col_name] = pdf[col_name].apply(_polars_map_entries_to_spark_dict)
        result_spark_df = spark.createDataFrame(pdf, schema=expected_spark_df.schema)
        assert_pyspark_df_equal(result_spark_df, expected_spark_df, ignore_nullable=True)

    def test_struct_requires_at_least_one_column(self):
        with pytest.raises(ValueError, match="struct requires at least one column"):
            struct()


class TestConcat:
    """Behaviour tests for :func:`~sparkleframe.polarsdf.functions.concat` (not copied from prior commits)."""

    def test_concat_without_inputs_raises(self) -> None:
        with pytest.raises(ValueError, match="concat requires at least one column"):
            concat()

    def test_concat_single_column_is_identity_on_strings(self) -> None:
        pl_df = pl.DataFrame({"token": ["zig", None, ""]})
        got = DataFrame(pl_df).select(concat("token").alias("out")).to_native_df()
        assert got["out"].to_list() == ["zig", None, ""]

    def test_concat_two_parts_any_null_yields_null(self) -> None:
        pl_df = pl.DataFrame(
            {
                "prefix": ["aa", None, "cc"],
                "suffix": ["bb", "bb", None],
            }
        )
        got = DataFrame(pl_df).select(concat(col("prefix"), col("suffix")).alias("out")).to_native_df()
        assert got["out"].to_list() == ["aabb", None, None]

    def test_concat_accepts_string_name_or_column_object(self) -> None:
        pl_df = pl.DataFrame({"segment": ["north", "south"]})
        sf = DataFrame(pl_df)
        by_name = sf.select(concat("segment", lit(":"), col("segment")).alias("out")).to_native_df()
        by_col = sf.select(concat(col("segment"), lit(":"), "segment").alias("out")).to_native_df()
        assert by_name["out"].to_list() == ["north:north", "south:south"]
        assert by_col["out"].to_list() == by_name["out"].to_list()

    def test_concat_coerces_integer_columns_like_strings(self, spark) -> None:
        pl_df = pl.DataFrame({"lane": [7, 0], "slot": [13, 42]})
        polars_df = DataFrame(pl_df)
        result_spark_df = create_spark_df(
            spark,
            polars_df.select(concat(col("lane"), lit("-"), col("slot")).alias("merged")),
        )
        spark_df = spark.createDataFrame(pl_df.to_pandas())
        expected_df = spark_df.select(
            spark_concat(spark_col("lane"), spark_lit("-"), spark_col("slot")).alias("merged"),
        )
        assert_pyspark_df_equal(result_spark_df, expected_df, ignore_nullable=True)


class TestTryToTimestamp:
    """Tests for try_to_timestamp — verifies null-safe parsing behaviour."""

    @pytest.mark.parametrize(
        "datetime_strs, fmt",
        [
            (["2023-01-01 12:34:56", "2024-02-02 23:45:01"], "yyyy-MM-dd HH:mm:ss"),
            (["01-03-2023 09:15:00", "31-12-2022 23:59:59"], "dd-MM-yyyy HH:mm:ss"),
            (["2024-05-31 20:14:19.993", "2023-12-12 11:11:11.123"], "yyyy-MM-dd HH:mm:ss.SSS"),
        ],
    )
    def test_try_to_timestamp_valid_matches_to_timestamp(self, spark, datetime_strs, fmt):
        df = pd.DataFrame({"ts": datetime_strs})
        polars_df = DataFrame(pl.DataFrame(df))

        result_strict = polars_df.select(to_timestamp("ts", fmt).alias("result")).to_native_df()
        result_try = polars_df.select(try_to_timestamp("ts", fmt).alias("result")).to_native_df()

        assert result_strict["result"].to_list() == result_try["result"].to_list()

    def test_try_to_timestamp_malformed_returns_null(self):
        df = pl.DataFrame({"ts": ["2023-01-01 12:34:56", "not-a-date", None]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_to_timestamp("ts").alias("result")).to_native_df()

        assert result["result"][0] is not None
        assert result["result"][1] is None
        assert result["result"][2] is None

    def test_try_to_timestamp_accepts_column_input(self):
        df = pl.DataFrame({"ts": ["2023-01-01 12:34:56"]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_to_timestamp(col("ts")).alias("result")).to_native_df()

        assert result["result"][0] is not None


class TestTryToDate:
    """Tests for try_to_date — verifies null-safe date parsing."""

    def test_try_to_date_default_format(self):
        df = pl.DataFrame({"d": ["1997-02-28", "2024-12-31", "bad", None]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_to_date("d").alias("result")).to_native_df()

        assert result["result"][0] == date(1997, 2, 28)
        assert result["result"][1] == date(2024, 12, 31)
        assert result["result"][2] is None
        assert result["result"][3] is None

    def test_try_to_date_custom_format(self):
        df = pl.DataFrame({"d": ["28-02-1997", "31-12-2024"]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_to_date("d", "dd-MM-yyyy").alias("result")).to_native_df()

        assert result["result"][0] == date(1997, 2, 28)
        assert result["result"][1] == date(2024, 12, 31)

    def test_try_to_date_accepts_column_input(self):
        df = pl.DataFrame({"d": ["2024-01-01"]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_to_date(col("d")).alias("result")).to_native_df()
        assert result["result"][0] is not None


class TestTryElementAt:
    """Tests for try_element_at — arrays (1-based) and maps."""

    def test_array_positive_index(self):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 1).alias("v")).to_native_df()
        assert result["v"][0] == "a"

    def test_array_last_element(self):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 3).alias("v")).to_native_df()
        assert result["v"][0] == "c"

    def test_array_negative_index(self):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", -1).alias("v")).to_native_df()
        assert result["v"][0] == "c"

    def test_array_oob_returns_null(self):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 4).alias("v")).to_native_df()
        assert result["v"][0] is None

    def test_array_zero_index_returns_null(self):
        df = pl.DataFrame({"arr": [["a", "b", "c"]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("arr", 0).alias("v")).to_native_df()
        assert result["v"][0] is None

    def test_map_key_present(self):
        df = pl.DataFrame({"m": [[{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("m", "a").alias("v")).to_native_df()
        assert result["v"][0] == 1.0

    def test_map_key_absent_returns_null(self):
        df = pl.DataFrame({"m": [[{"key": "a", "value": 1.0}, {"key": "b", "value": 2.0}]]})
        polars_df = DataFrame(df)
        result = polars_df.select(try_element_at("m", "c").alias("v")).to_nativ

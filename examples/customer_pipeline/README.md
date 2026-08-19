# Example: customer revenue pipeline

A legacy pandas ETL that computes total order revenue per country for adult
customers. It exists to be migrated — and, more importantly, to be migrated
*wrongly* by anything that treats the job as text translation.

## The pipeline

```python
customers = pd.read_csv(...)
orders    = pd.read_csv(...)

customers = customers[customers["age"] >= 18]
orders["revenue"] = orders["quantity"] * orders["price"]

result = customers.merge(orders, on="customer_id", how="left")
result = result.groupby("country")["revenue"].sum().reset_index()
result = result.sort_values(by="country")
result.to_csv(..., index=False)
```

## The traps

Each one is planted deliberately, activated by the generated data, and asserted
on in `tests/`.

| # | Trap | Naive Spark result |
|---|---|---|
| 1 | `age >= 18` on a column with nulls | Matches by luck — `>=` drops null in both engines. Would *not* match for `!=`. |
| 2 | `age` profiles as `float64`, not `int64` | `inferSchema` picks `IntegerType`; schema comparison fails. |
| 3 | LEFT join leaves ~400 customers with null revenue | Fine, but it makes trap 5 reachable. |
| 4 | **`groupby("country")` with 63 null countries** | pandas drops them (`dropna=True`); Spark keeps null as its own group. **One extra row, different totals.** |
| 5 | `.sum()` over an all-NaN group | pandas returns `0.0`; Spark returns `null`. |
| 6 | `reset_index()` | No Spark equivalent. Synthesising an index column breaks schema parity. |

Trap 4 is the headline. Here is what a naive translation actually produces:

```text
+-------+------------------+
|country|revenue           |
+-------+------------------+
|NULL   |143967.80999999997|   <-- phantom row, 143,967.81 of revenue
|DE     |891697.5199999994 |
|ES     |913905.3000000007 |
|FR     |787478.2199999993 |
|MA     |798452.6399999997 |
|US     |859786.1599999995 |
+-------+------------------+
6 rows — the legacy pipeline produced 5.
```

No exception, no warning. A plausible-looking report with a wrong number in it.
That is the failure mode this whole system is built to catch.

## Running it

```bash
# Regenerate the input data (seeded, so it is byte-identical everywhere)
python examples/customer_pipeline/generate_data.py --rows 2000

# Migrate: discovery -> planning -> PySpark -> static gate
etl-migrator migrate examples/customer_pipeline/legacy_pipeline.py

# The reference implementation on its own
python examples/customer_pipeline/legacy_pipeline.py \
    examples/customer_pipeline/input /tmp/legacy_out
```

## What the system produces

The plan flags trap 4 as `HIGH` risk, which triggers the human-approval gate,
and declares a `SemanticDifference` for each divergence — every one of which
becomes a mandatory validation check. The generated PySpark then implements the
mitigations:

```python
revenue_by_country = (
    joined.filter(F.col("country").isNotNull())        # trap 4: pandas dropna=True
    .groupBy("country")
    .agg(F.coalesce(F.sum("revenue"), F.lit(0.0))      # trap 5: all-null sum -> 0.0
         .alias("revenue"))                            # Spark would name it sum(revenue)
)
```

Measured equivalence against the legacy output:

```text
schema        : MATCH  ['country', 'revenue']
row count     : 5 vs 5
group keys    : MATCH
max rel diff  : 8.87e-16   (tolerance 1e-6)
null counts   : 0 vs 0
```

`tests/test_spark_equivalence.py` runs exactly this comparison, plus a
counterfactual test that fails if trap 4 ever stops being exercised.

## Files

```text
legacy_pipeline.py    the pipeline to migrate, with traps documented inline
generate_data.py      seeded generator; the distribution activates every trap
input/                customers.csv (2000 rows), orders.csv (4013 rows)
```

`input/` is committed so the example runs immediately after clone. Regenerating
with the same seed reproduces it byte for byte.

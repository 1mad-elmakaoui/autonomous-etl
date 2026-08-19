"""Legacy customer revenue ETL.

Written the way real legacy pipelines are written: pandas, in-place mutation,
implicit dtype handling, and a couple of behaviours that are load-bearing
without being obvious. It is the migration system's job to notice those.

The traps deliberately planted here, in increasing order of subtlety:

1. `age >= 18` on a column that contains nulls — pandas drops NaN rows silently.
2. `quantity * price` where price is float and quantity is int — result dtype.
3. A LEFT join, so customers with no orders carry null revenue downstream.
4. `groupby("country")` where some customers have a null country — pandas drops
   those rows by default (dropna=True); Spark keeps null as its own group. This
   single default is the most common source of silent row-count differences in
   pandas-to-Spark migrations.
5. `.sum()` over a group that is entirely NaN returns 0.0 in pandas but null in
   Spark.

The pipeline exposes `run(input_dir, output_dir)` so the harness can execute it
as the reference implementation without importing anything at module scope.
"""

import os

import pandas as pd

MIN_AGE = 18


def run(input_dir, output_dir):
    customers = pd.read_csv(os.path.join(input_dir, "customers.csv"))
    orders = pd.read_csv(os.path.join(input_dir, "orders.csv"))

    customers = customers[customers["age"] >= MIN_AGE]

    orders["revenue"] = orders["quantity"] * orders["price"]

    result = customers.merge(orders, on="customer_id", how="left")

    result = result.groupby("country")["revenue"].sum().reset_index()

    result = result.sort_values(by="country")

    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, "revenue_by_country.csv"), index=False)


if __name__ == "__main__":
    import sys

    run(sys.argv[1], sys.argv[2])

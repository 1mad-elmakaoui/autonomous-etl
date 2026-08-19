"""Deterministic input-data generator for the customer example.

Seeded, so the same rows are produced on every machine and in CI — the whole
validation story depends on the reference output being reproducible.

The distribution is chosen to *activate* the traps documented in
legacy_pipeline.py rather than to look realistic: null ages, null countries,
customers with no orders, and one country whose only revenue rows are null.

    python examples/customer_pipeline/generate_data.py --rows 2000
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

COUNTRIES = ["FR", "DE", "MA", "US", "ES"]
HERE = Path(__file__).parent


def generate(customer_rows: int, seed: int, out_dir: Path) -> tuple[Path, Path]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    customers_path = out_dir / "customers.csv"
    orders_path = out_dir / "orders.csv"

    customer_ids: list[int] = []
    with customers_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("customer_id,name,age,country,signup_date\n")
        for cid in range(1, customer_rows + 1):
            customer_ids.append(cid)
            # ~4% null ages, ~3% null countries: both traps, both realistic.
            age = "" if rng.random() < 0.04 else str(rng.randint(12, 89))
            country = "" if rng.random() < 0.03 else rng.choice(COUNTRIES)
            day = rng.randint(1, 28)
            month = rng.randint(1, 12)
            fh.write(f"{cid},customer_{cid},{age},{country},2024-{month:02d}-{day:02d}\n")

    with orders_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("order_id,customer_id,product,quantity,price,order_date\n")
        order_id = 1
        for cid in customer_ids:
            # ~18% of customers have no orders at all -> null revenue after the LEFT join.
            if rng.random() < 0.18:
                continue
            for _ in range(rng.randint(1, 4)):
                quantity = rng.randint(1, 9)
                price = round(rng.uniform(4.5, 499.99), 2)
                day = rng.randint(1, 28)
                month = rng.randint(1, 12)
                fh.write(
                    f"{order_id},{cid},product_{rng.randint(1, 60)},"
                    f"{quantity},{price},2024-{month:02d}-{day:02d}\n"
                )
                order_id += 1

    return customers_path, orders_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=2000, help="number of customers")
    parser.add_argument("--seed", type=int, default=20240215)
    parser.add_argument("--out", type=Path, default=HERE / "input")
    args = parser.parse_args()

    customers, orders = generate(args.rows, args.seed, args.out)
    for path in (customers, orders):
        lines = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"wrote {path}  ({lines} rows)")


if __name__ == "__main__":
    main()

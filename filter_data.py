#!/usr/bin/env python3
import csv
import argparse
from decimal import Decimal, InvalidOperation


def matches(value, target_str):
    if value is None:
        return False
    v = value.strip()
    if v == target_str:
        return True
    try:
        return Decimal(v) == Decimal(target_str)
    except (InvalidOperation, ValueError):
        try:
            return abs(float(v) - float(target_str)) < 1e-12
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(description="Filter CSV rows by total_samples_planned value")
    parser.add_argument("-i", "--input", default="scan_benchmark/vlm/data.csv", help="Input CSV path")
    parser.add_argument("-o", "--output", default="filtered_data.csv", help="Output CSV path")
    parser.add_argument("-v", "--value", default="24473395", help="Value to match for total_samples_planned")
    args = parser.parse_args()

    with open(args.input, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            print(f"No header found in {args.input}")
            return
        rows = [r for r in reader if matches(r.get('total_samples_planned'), args.value)]

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == '__main__':
    main()

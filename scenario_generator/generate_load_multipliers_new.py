"""
Download and read distribution system load curves from the NREL SMART-DS dataset
hosted on the OEDI AWS S3 bucket.

Dataset: https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=SMART-DS%2Fv1.0%2F
Bucket: oedi-data-lake (public, anonymous access)
Region: us-west-2

Dependencies: boto3, botocore, pandas, matplotlib, tqdm
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BUCKET = "oedi-data-lake"
SMARTDS_PREFIX = "SMART-DS/v1.0/"
AWS_REGION = "us-west-2"


def get_s3_client():
    """Create an anonymous S3 client for the public OEDI bucket."""
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        config=Config(signature_version=UNSIGNED),
    )


def list_s3_objects(bucket: str, prefix: str) -> List[str]:
    """Paginate through an S3 prefix and return all object keys.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to list under.

    Returns:
        List of full S3 keys found under the prefix.
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []

    logger.info("Listing objects under s3://%s/%s", bucket, prefix)
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    logger.info("Found %d objects.", len(keys))
    return keys


def list_common_prefixes(bucket: str, prefix: str, delimiter: str = "/") -> List[str]:
    """List 'subdirectories' (common prefixes) under a given S3 prefix.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to list under.
        delimiter: Delimiter for grouping keys (default '/').

    Returns:
        List of common prefix strings (subdirectory-like paths).
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    prefixes: List[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter=delimiter):
        for cp in page.get("CommonPrefixes", []):
            prefixes.append(cp["Prefix"])

    return prefixes


def download_file(bucket: str, key: str, local_path: str) -> None:
    """Download a single file from S3 to a local path.

    Args:
        bucket: S3 bucket name.
        key: Full S3 object key.
        local_path: Local filesystem path to save the file.

    Raises:
        FileNotFoundError: If the S3 key does not exist.
    """
    s3 = get_s3_client()
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(f"S3 key not found: s3://{bucket}/{key}") from e
        raise

    logger.debug("Downloading s3://%s/%s -> %s", bucket, key, local_path)
    s3.download_file(bucket, key, local_path)


def load_profile_to_dataframe(
    local_csv_path: str, year: int = 2018, freq: str = "h"
) -> pd.DataFrame:
    """Load a SMART-DS load profile CSV into a pandas DataFrame with a datetime index.

    Args:
        local_csv_path: Path to the local CSV file.
        year: Year for the datetime index (default 2018).
        freq: Frequency string for the datetime index ('h' for hourly, '15min' for 15-min).

    Returns:
        DataFrame with a DatetimeIndex and load profile columns.
    """
    df = pd.read_csv(local_csv_path)

    # Determine number of rows to infer resolution if not specified
    n_rows = len(df)
    if n_rows == 8760:
        freq = "h"
    elif n_rows == 35040:
        freq = "15min"

    start = pd.Timestamp(f"{year}-01-01 00:00:00")
    time_index = pd.date_range(start=start, periods=n_rows, freq=freq)
    df.index = time_index
    df.index.name = "timestamp"

    return df


def build_feeder_prefix(
    year: str, region: str, substation: str, scenario: str, feeder: Optional[str] = None
) -> str:
    """Construct the S3 prefix for a given feeder path.

    Args:
        year: Dataset year (e.g. '2018').
        region: Region code (e.g. 'AUS').
        substation: Substation identifier.
        scenario: Scenario name (e.g. 'base_timeseries').
        feeder: Feeder name. If None, prefix stops at opendss/.

    Returns:
        S3 prefix string.
    """
    prefix = f"{SMARTDS_PREFIX}{year}/{region}/{substation}/scenarios/{scenario}/opendss/"
    if feeder:
        prefix += f"{feeder}/"
    return prefix


def main() -> None:
    """CLI entry point: download and plot SMART-DS load profiles."""
    parser = argparse.ArgumentParser(
        description="Download SMART-DS load profiles from OEDI S3 bucket."
    )
    parser.add_argument("--year", type=str, default="2018", help="Dataset year.")
    parser.add_argument("--region", type=str, default="AUS", help="Region code.")
    parser.add_argument(
        "--substation", type=str, default="p1uhs0_1247", help="Substation identifier."
    )
    parser.add_argument(
        "--scenario", type=str, default="base_timeseries", help="Scenario name."
    )
    parser.add_argument("--feeder", type=str, default=None, help="Feeder name (auto-detected if omitted).")
    parser.add_argument(
        "--output-dir", type=str, default="data", help="Local directory for downloaded files."
    )
    args = parser.parse_args()

    # --- Show available years and regions ---
    logger.info("Available years/regions under SMART-DS/v1.0/:")
    years = list_common_prefixes(BUCKET, SMARTDS_PREFIX)
    for y in years:
        regions = list_common_prefixes(BUCKET, y)
        region_names = [r.rstrip("/").split("/")[-1] for r in regions]
        logger.info("  %s -> regions: %s", y.rstrip("/").split("/")[-1], region_names)

    # --- Resolve feeder ---
    opendss_prefix = build_feeder_prefix(args.year, args.region, args.substation, args.scenario)
    if args.feeder:
        feeder_prefix = opendss_prefix + args.feeder + "/"
    else:
        feeders = list_common_prefixes(BUCKET, opendss_prefix)
        if not feeders:
            logger.error("No feeders found under %s", opendss_prefix)
            return
        feeder_prefix = feeders[0]
        logger.info("Auto-selected feeder: %s", feeder_prefix.rstrip("/").split("/")[-1])

    # --- Find and download profile CSVs ---
    profiles_prefix = feeder_prefix + "profiles/"
    all_keys = list_s3_objects(BUCKET, profiles_prefix)
    csv_keys = [k for k in all_keys if k.lower().endswith(".csv")]

    if not csv_keys:
        logger.warning("No CSV profiles found under %s", profiles_prefix)
        return

    logger.info("Downloading %d profile CSV(s) to '%s'...", len(csv_keys), args.output_dir)
    local_paths: List[str] = []
    for key in tqdm(csv_keys, desc="Downloading profiles"):
        relative = key[len(profiles_prefix):]
        local_path = os.path.join(args.output_dir, relative)
        try:
            download_file(BUCKET, key, local_path)
            local_paths.append(local_path)
        except FileNotFoundError:
            logger.warning("Key not found, skipping: %s", key)

    if not local_paths:
        logger.error("No files were downloaded successfully.")
        return

    # --- Load and plot one profile ---
    sample_path = local_paths[0]
    logger.info("Loading profile: %s", sample_path)
    df = load_profile_to_dataframe(sample_path, year=int(args.year))
    logger.info("Profile shape: %s, columns: %s", df.shape, list(df.columns))

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    # Plot first numeric column
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        logger.error("No numeric columns found in %s", sample_path)
        return

    col = numeric_cols[0]
    ax.plot(df.index, df[col], linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel(col)
    ax.set_title(f"SMART-DS Load Profile: {Path(sample_path).stem}")
    fig.tight_layout()

    plot_path = os.path.join(args.output_dir, "profile_plot.png")
    fig.savefig(plot_path, dpi=150)
    logger.info("Plot saved to %s", plot_path)
    plt.show()


if __name__ == "__main__":
    main()

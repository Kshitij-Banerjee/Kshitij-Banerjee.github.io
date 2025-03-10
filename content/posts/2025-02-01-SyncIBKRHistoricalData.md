---
Category: ML Trading
Title: Syncing historical data from IBKR
Layout: post
Name: Syncing historical data from IBKR
date: 2025-02-01
tags: [ML, Trading, AI, Beginner, AlgorthmicTrading, Basics]
keywords:
  [
    ML,
    machine-learning,
    AI,
    Transformers,
    Trading,
    AlgorithmicTrading,
    Beginner,
    Basics,
  ]
---

# Syncing Historical Data from IBKR: A Comprehensive Guide

In this post, we'll walk through a complete workflow for downloading historical data from Interactive Brokers (IBKR) and preparing it for analysis and backtesting.

## Why download from broker ?

The core assumption is that we sync data directly from the broker, ensuring its accuracy while trading and backtesting.

Once this data is downloaded, we can build ML data batches for training models.

We can also backtest trading strategies using tools like Backtrader on the same data.

Added bonus: It's all (mostly) free apart from the market data subscription that you would likely anyways be paying.

## Strategy

Our strategy involves

1. Downloading data in 1-minute candles,

2. Resampling it into higher timeframes (15 minutes, 1 day, and 1 week),

3. Storing it locally, and 3

4. Finally syncing it to a cloud bucket (e.g., Google Cloud) for backup.

## Overview

So we'll cover the following core components:-

**Setting up TWS API**: The brokers API instance settings

**Broker Data Sync:** Download historical data directly from IBKR using their API.

**Data Resampling:** Convert the 1-minute candles into higher-level candles (15m, 1d, 1w) to suit various analysis needs.

**Local Storage & Cloud Backup:** Save data locally using a structured directory system and use `rsync` to copy the data to a cloud bucket.

# Setting Up the TWS API for Paper Trading

To download historical data from IBKR, you need to set up the Trader Workstation (TWS) or IB Gateway API. Here’s how you can configure it for paper trading:

### 1. Install IBKR Trader Workstation (TWS)

Download and install [TWS](https://interactivebrokers.github.io/) from the IBKR website.

### 2. Enable API Access

Open TWS and log in with your **paper trading account**.

Navigate to **Edit** → **Global Configuration**.

Under **API** → **Settings**, enable:

☑ _Enable ActiveX and Socket Clients_

☑ _Allow connections from localhost only_ (for security)

☑ _Read-Only API_ (optional for safety)

Set **Socket Port** to `7497` (default for paper trading).

### 3. Configure Market Data Subscription

Ensure that your IBKR account has **market data subscriptions** active. Without it, historical data requests may be blocked.

### 4. Run the API Client

TWS must **remain open** while your script runs. If you prefer a headless setup, use **IB Gateway** instead.

Now, you’re ready to connect your script to IBKR’s paper trading environment!

# Downloading Historical Data from IBKR

**Step 1:** We start with a shell script that sets up the environment, creates the necessary directory structure (using placeholders for your preferred paths), downloads the data with proper rate limiting, and finally syncs the data to the cloud.

### Shell Script: Historical Data Downloader

```bash
#!/bin/bash

# Parse arguments
TICKER=$1
START_DATE=$2
END_DATE=$3

# Configuration
BUCKET_NAME="your-cloud-bucket"  # Replace with your actual bucket name
LOCAL_STORAGE=" $HOME/your/local/storage/path/$ TICKER"  # Use your placeholder path
PYTHON_SCRIPT="$HOME/your/python/script/path/ibkr_historical_downloader.py"  # Use your placeholder path

# Parse arguments and convert to date format
START_DATE= $(date -j -f "%Y-%m-%d" "$ START_DATE" "+%Y-%m-%d")
END_DATE= $(date -j -f "%Y-%m-%d" "$ END_DATE" "+%Y-%m-%d")
CURRENT_DATE="$START_DATE"

# Create date partitions
while  "$CURRENT_DATE" < "$END_DATE" || "$CURRENT_DATE" == "$END_DATE" ; do
	YEAR= $(date -j -f "%Y-%m-%d" "$ CURRENT_DATE" "+%Y")
	MONTH= $(date -j -f "%Y-%m-%d" "$ CURRENT_DATE" "+%m")
	mkdir -p " $LOCAL_STORAGE/$ YEAR/$MONTH"
	# Increment the date by one day
	CURRENT_DATE= $(date -j -f "%Y-%m-%d" "$ CURRENT_DATE" -v+1d "+%Y-%m-%d")
done

# Run data download with rate limiting
python3 " $PYTHON_SCRIPT" --ticker "$ TICKER" --start-date " $START_DATE" --end-date "$ END_DATE"

# Cloud Sync with checksum validation
echo "Starting cloud sync..."
gsutil -m rsync -d -r " $LOCAL_STORAGE" "gs://$ BUCKET_NAME/$TICKER"
echo "Sync complete. Verify with: gsutil ls gs://$BUCKET_NAME/$TICKER"
```

### Python CLI for connecting to IBKR Trader Workstation and downloading

```python
#!/usr/bin/env python3

import argparse
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
import pytz
import os

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.data_event = threading.Event()
        self.request_complete = False

    def historicalData(self, reqId, bar):
        try:
            # Parse timestamp with timezone
            date_str, tz_str = bar.date.split(' US/')
            naive_dt = datetime.strptime(date_str, "%Y%m%d %H:%M:%S")
            tz = pytz.timezone(f'US/{tz_str}')
            localized_dt = tz.localize(naive_dt)
            utc_dt = localized_dt.astimezone(pytz.utc)

            self.data.append({
                'timestamp': utc_dt,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume
            })
        except Exception as e:
            print(f"Error processing bar: {e}")

    def historicalDataEnd(self, reqId, start, end):
        self.request_complete = True
        self.data_event.set()

    def run_loop(self):
        self.run()

def main(args):
    # Initialize API
    app = IBapi()
    app.connect('127.0.0.1', 7497, clientId=2)

    # Connection timeout handling
    connect_timeout = 10
    start_time = time.time()
    while not app.isConnected():
        if time.time() - start_time > connect_timeout:
            raise Exception("Connection timeout")
        time.sleep(0.1)

    # Start API thread
    api_thread = threading.Thread(target=app.run_loop, daemon=True)
    api_thread.start()
    time.sleep(1)  # Stabilization period

    # Contract setup
    contract = Contract()
    contract.symbol = args.ticker
    contract.secType = 'STK'
    contract.exchange = 'SMART'
    contract.currency = 'USD'

    # Date handling
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Loop through requests
    current_date = start_date
    while current_date <= end_date:
        end_date_str = current_date.strftime("%Y%m%d 23:59:59 US/Eastern")
        next_date = current_date + timedelta(days=1)

        # Request parameters
        app.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=end_date_str,
            durationStr='1 D',
            barSizeSetting='1 min',
            whatToShow='TRADES',
            useRTH=0,  # Use all available data, not just Regular Trading Hours
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

        # Wait for data with timeout
        if not app.data_event.wait(30):
            print(f"Timeout waiting for data on {current_date.strftime('%Y-%m-%d')}")
            return

        # Process and save data
        if app.data:
            df = pd.DataFrame(app.data)
            df.set_index('timestamp', inplace=True)

            # Create directory structure
            year = current_date.strftime("%Y")
            month = current_date.strftime("%m")
            filename = f"{current_date.strftime('%Y-%m-%d')}.parquet.snappy"
            storage_dir = f"{os.environ['HOME']}/your/local/storage/path/{args.ticker}/{year}/{month}"
            os.makedirs(storage_dir, exist_ok=True)

            # Save with partitioning
            df.to_parquet(
                os.path.join(storage_dir, filename),
                compression='snappy',
                index=True
            )
            print(f"Saved {len(df)} rows for {current_date.strftime('%Y-%m-%d')}")

            # Clear data for next request
            app.data = []
            app.data_event.clear()
        else:
            print("NO DATA!")

        current_date = next_date

    app.disconnect()
    api_thread.join(timeout=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Historical Data Downloader')
    parser.add_argument('--ticker', type=str, required=True,
                      help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, required=True,
                      help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, required=True,
                      help='End date in YYYY-MM-DD format')
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Fatal error: {e}")
        exit(1)

```

## Loading and Resampling Data

After downloading the minute-level data, you might want to resample the candles into higher-level intervals (15 minutes, 1 day, and 1 week).

Here are two helper functions: one for loading the Parquet files and another for resampling the candlestick data.

### 1. Loading Parquet Files

```python
import os
import pandas as pd
from typing import List

def load_parquet_files(directory: str) -> pd.DataFrame:
  """
  Loads and concatenates all Parquet files in the specified directory,
  assuming filenames start with a date in 'YYYY-MM-DD' format.

  Args:
      directory (str): Path to the directory containing Parquet files.

  Returns:
      pd.DataFrame: A concatenated DataFrame of all Parquet files sorted by index.
  """
  data = []
  file_paths: List[tuple[str, str]] = []

  # Walk through directories and collect the file paths with their associated dates
  for root, _, files in os.walk(directory):
      for file in files:
          if file.endswith(".parquet.snappy"):
              file_path = os.path.join(root, file)
              date_str = file.split('_')[0]  # Modify if the date is in a different part of the filename
              file_paths.append((file_path, date_str))

  # Sort file paths by the date
  file_paths.sort(key=lambda x: x[1])

  # Read the files in order of their dates
  for file_path, _ in file_paths:
      print(f"Reading file: {file_path}")
      df = pd.read_parquet(file_path, engine='pyarrow')
      data.append(df)

  # Concatenate all the data into a single DataFrame
  if data:
      full_df = pd.concat(data, ignore_index=False)
      full_df = full_df.sort_index().drop_duplicates()
      print("Data loaded into DataFrame successfully.")
      return full_df
  else:
      print("No Parquet files found in the directory.")
      return pd.DataFrame()

# Example usage
if __name__ == "__main__":
  LOCAL_STORAGE = os.path.expanduser("~/your/local/storage/path/TSLA/2025/01/")
  df = load_parquet_files(LOCAL_STORAGE)
  print(df.head())
```

### 2. Resampling Candlestick Data

```python
from typing import Literal
import pandas as pd
import argparse

def resample_candles(df: pd.DataFrame, interval: Literal['15min', '1D', '1W']) -> pd.DataFrame:
  """
  Resamples minute-level candlestick data to higher timeframes.
  IMP: The input data is expected to be a minute-level candle

  Args:
      df (pd.DataFrame): The input DataFrame with minute-level data.
      interval (Literal['15min', '1D', '1W']): The resampling interval.

  Returns:
      pd.DataFrame: The resampled DataFrame.
  """
  resampled_df = df.resample(interval).agg({
      'open': 'first',
      'high': 'max',
      'low': 'min',
      'close': 'last',
      'volume': 'sum'
  }).dropna()
  return resampled_df

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Resample candlestick data.")
  parser.add_argument("file", type=str, help="Path to the Parquet file containing minute-level data.")
  parser.add_argument("interval", type=str, choices=['15min', '1D', '1W'], help="Resampling interval.")
  args = parser.parse_args()

  df = pd.read_parquet(args.file)
  df.index = pd.to_datetime(df.index)  # Ensure index is datetime

  resampled_df = resample_candles(df, args.interval)
  print(resampled_df.head())
```

#### Example Command

```bash
python ./resample_candles.py ./your/data/path/SPY/2024/01/2024-01-01.parquet.snappy 15min
```

Expected output:

```yaml
open    high     low   close volume
timestamp
2023-12-29 09:00:00+00:00  477.40  477.46  477.37  477.37   2786
2023-12-29 09:15:00+00:00  477.37  477.37  477.34  477.34    100
2023-12-29 09:30:00+00:00  477.34  477.34  477.24  477.28   2638
2023-12-29 09:45:00+00:00  477.28  477.28  477.11  477.11  13678
2023-12-29 10:00:00+00:00  477.12  477.18  477.07  477.13  12245
```

# Conclusion

This setup provides a robust framework for ensuring that your historical data is accurate, readily available, and formatted as needed for various use cases—be it training ML models or backtesting trading strategies.

With the flexibility of resampling and the reliability of syncing data to the cloud, you can be assured that your data pipeline will support your trading operations and research effectively.

Happy coding and trading!

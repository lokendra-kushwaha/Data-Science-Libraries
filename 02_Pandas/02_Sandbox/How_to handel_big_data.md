# 🚀 How to Handle 1 Billion Rows in Pandas

When dealing with massive datasets (e.g., 1 Billion rows), loading a CSV directly via `pd.read_csv()` will cause an Out of Memory (OOM) error because a standard machine does not have enough RAM (50-100 GB) to hold it all at once. To handle this, Data Engineers use the following 4 core strategies:

## 1. Chunking (Divide and Conquer)
Instead of loading all data at once, we tell Pandas to read the file in smaller "chunks" (e.g., 100,000 rows at a time). We process each chunk, save the intermediate result, and immediately discard the chunk from RAM.

```python
import pandas as pd

# Processing 100,000 rows at a time
chunk_size = 100_000
total_sales = 0

# pd.read_csv returns an iterator when chunksize is provided
for chunk in pd.read_csv('massive_1B_data.csv', chunksize=chunk_size):
    # Perform calculation on the current chunk
    total_sales += chunk['sales_amount'].sum()
    
print(f"Total Sales across 1 Billion rows: {total_sales}")
```

## 2. Data Type Optimization (Downcasting)
Pandas assigns maximum memory by default. An integer is saved as `int64` (8 bytes), even if the number is just '25'. 
*   **Numbers:** Downcast `int64` to `int32` or `int8` if the values are small.
*   **Strings:** Pandas stores strings as `object` (which is very heavy). If a column has repetitive strings (like 'Male'/'Female' or 'City Names'), convert it to `category`. This single trick reduces string memory usage by up to 90%.

```python
# Optimizing during the loading phase
optimized_dtypes = {
    'age': 'int8',          # Takes 1 byte instead of 8 bytes
    'gender': 'category',   # Uses mapping (0, 1) instead of storing raw strings
    'city': 'category'      # Extremely efficient for repetitive categorical data
}

df = pd.read_csv('massive_data.csv', dtype=optimized_dtypes)
```

## 3. Drop the CSV Format (Use Parquet)
CSV is the worst format for big data. It is uncompressed, slow, and row-based.
In production environments, we convert CSVs to **Parquet**. Parquet is a columnar storage format that compresses data heavily and allows Pandas to load only specific columns incredibly fast without reading the entire file into memory.

```python
# Reading specific columns from a Parquet file is instantly fast and memory-efficient
df = pd.read_parquet('massive_data.parquet', columns=['age', 'salary'])
```

## 4. Bypassing Pandas (Polars / Dask)
If the data is truly massive and requires parallel processing (using all CPU cores simultaneously), Pandas reaches its architectural limits because it runs strictly on a single core. In such cases, we switch to:
*   **Polars:** A modern, blazing-fast DataFrame library written in Rust that utilizes all CPU cores.
*   **Dask:** A parallel computing library that scales Pandas syntax across multiple cores or even multiple machines/servers.
# %%
from pathlib import Path

import pandas as pd

# Directory to search for .YMR files
search_dir = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\data_vanCarlijn\CNonequi\CNonequi"

# Find all .YMR files
ymr_files = list(Path(search_dir).glob("*.YMR"))
print(f"Found {len(ymr_files)} .YMR files")

# Concatenate all files into a single dataframe
dfs = []
for file in ymr_files:
    try:
        # Read the file, skipping header lines until we find DATUM
        with open(file, "r") as f:
            lines = f.readlines()
            first_line = lines[0].strip()
            # Find the line with "DATUM"
            header_idx = None
            for i, line in enumerate(lines):
                if "DATUM" in line:
                    header_idx = i
                    break

            if header_idx is not None:
                # Read from DATUM line onwards
                df = pd.read_csv(
                    file, sep="\t", skiprows=header_idx, na_values=["#NTB"]
                )
                df.columns = df.columns.str.replace("*", "", regex=False)
                df["WAARDE"] = pd.to_numeric(df["WAARDE"], errors="coerce")
                df["DATUM"] = pd.to_datetime(df["DATUM"], format="%Y%m%d")
                df["NAME"] = first_line
                df["FILE"] = file.name
                dfs.append(df)
                print(f"Loaded: {file.name} - {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file.name}: {e}")

# Concatenate all dataframes
if dfs:
    result_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataframe shape: {result_df.shape}")
    print(result_df.head())

    # Optional: Save to CSV
    result_df.to_csv("combined_neerslag.csv", index=False)
    print("\nSaved to combined_neerslag.csv")
else:
    print("No files were loaded successfully")

result_df
# %%
# Plot WAARDE per DATUM for each FILE
if dfs:
    import matplotlib.pyplot as plt

    files = result_df["NAME"].unique()
    for file in files:
        file_data = (
            result_df[result_df["NAME"] == file]
            .sort_values("DATUM")
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            file_data["DATUM"],
            file_data["WAARDE"],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label="Daily",
        )
        ax.set_xlabel("")
        ax.set_ylabel("Chloride gehalte in mg/l", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        plt.title(f"Chloride gehalte in mg/l - {file}")
        plt.xticks(rotation=45)
        plt.xlim(pd.Timestamp(2017, 12, 31), file_data["DATUM"].max())
        ax.grid()
        plt.tight_layout()
        plt.show()
# %%
# %%
# Plot WAARDE per DATUM for each FILE
# %%
# Plot monthly max and min WAARDE per location
if dfs:
    import matplotlib.pyplot as plt

    files = result_df["NAME"].unique()
    for file in files:
        file_data = (
            result_df[result_df["NAME"] == file]
            .sort_values("DATUM")
            .reset_index(drop=True)
        )
        file_data["Month"] = file_data["DATUM"].dt.month

        monthly_stats = (
            file_data.groupby("Month")["WAARDE"].agg(["min", "max"]).reset_index()
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(
            monthly_stats["Month"],
            monthly_stats["max"],
            marker="o",
            linestyle="-",
            color="tab:red",
            label="Max",
        )
        ax.plot(
            monthly_stats["Month"],
            monthly_stats["min"],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label="Min",
        )
        ax.fill_between(
            monthly_stats["Month"],
            monthly_stats["min"],
            monthly_stats["max"],
            alpha=0.2,
            color="tab:gray",
        )

        ax.set_xlabel("Month of Year")
        ax.set_ylabel("Chloride gehalte in mg/l")
        ax.legend()

        plt.title(f"Monthly Min/Max Chloride gehalte in mg/l - {file}")
        ax.grid()
        plt.tight_layout()
        plt.show()

# %%
# Plot daily max and min WAARDE per location (by day of year)
if dfs:
    import matplotlib.pyplot as plt

    files = result_df["NAME"].unique()
    for file in files:
        file_data = (
            result_df[result_df["NAME"] == file]
            .sort_values("DATUM")
            .reset_index(drop=True)
        )
        file_data["4WeeksOfYear"] = file_data["DATUM"].dt.isocalendar().week // 4 + 1

        weekly_stats = (
            file_data.groupby("4WeeksOfYear")["WAARDE"]
            .agg(["min", "max"])
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            weekly_stats["4WeeksOfYear"],
            weekly_stats["max"],
            marker="o",
            linestyle="-",
            color="tab:red",
            label="Max",
            markersize=3,
        )
        ax.plot(
            weekly_stats["4WeeksOfYear"],
            weekly_stats["min"],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label="Min",
            markersize=3,
        )
        ax.fill_between(
            weekly_stats["4WeeksOfYear"],
            weekly_stats["min"],
            weekly_stats["max"],
            alpha=0.2,
            color="tab:gray",
        )

        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Chloride gehalte in mg/l")
        ax.legend()

        plt.title(f"Weekly Min/Max Chloride gehalte in mg/l - {file}")
        ax.set_xlim(0, 15)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
# %%
file_data
# %%

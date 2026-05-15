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
                df = df[df["DATUM"] > pd.to_datetime("2018-01-01")]
                df["NAME"] = first_line
                df["FILE"] = file.name
                dfs.append(df)
                print(f"Loaded: {file.name} - {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file.name}: {e}")

# Concatenate all dataframes
if dfs:
    result_df_sal = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataframe shape: {result_df_sal.shape}")
    print(result_df_sal.head())
else:
    print("No files were loaded successfully")
result_df_sal # mg/l
# %%
from pathlib import Path

import pandas as pd

# Directory to search for .YMR files
search_dir = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\data_vanCarlijn\Qnet.2018_2022\Qnet"

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
                df = df[df["DATUM"] > pd.to_datetime("2018-01-01")]
                df["NAME"] = first_line
                df["FILE"] = file.name
                dfs.append(df)
                print(f"Loaded: {file.name} - {len(df)} rows")
    except Exception as e:
        print(f"Error loading {file.name}: {e}")

# Concatenate all dataframes
if dfs:
    result_df_deb = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataframe shape: {result_df_deb.shape}")
    print(result_df_deb.head())
else:
    print("No files were loaded successfully")
result_df_deb # Q in miljoen m3 per dag
# %%
merged_df = pd.merge(result_df_sal, result_df_deb, on=["DATUM", "NAME"], suffixes=("_sal", "_deb"))
merged_df["WAARDE"] = (merged_df["WAARDE_sal"] * merged_df["WAARDE_deb"]) * 1000 
# Convert mg/l and million m3/day to kg/day
merged_df = merged_df[merged_df["NAME"] != "Berging IJsselmeer"]
merged_df
# %%
# Extract year from DATUM and sum WAARDE by location and year
merged_df["YEAR"] = merged_df["DATUM"].dt.year.astype(int)
yearly_summary = merged_df.groupby(["NAME", "YEAR"])["WAARDE"].sum().reset_index()
yearly_summary.columns = ["LOCATION", "YEAR", "WAARDE_SUM"]
yearly_summary = yearly_summary[yearly_summary["LOCATION"] != "Berging IJsselmeer"]
yearly_summary
# %%
import matplotlib.pyplot as plt

# Pivot to have years as columns
yearly_pivot = yearly_summary.pivot(index="LOCATION", columns="YEAR", values="WAARDE_SUM")
print(yearly_pivot)

# Plot years as x-axis for each unique location as subplot
locations = yearly_summary["LOCATION"].unique()
fig, axes = plt.subplots(len(locations), 1, figsize=(12, 4 * len(locations)), sharey=True)

if len(locations) == 1:
    axes = [axes]

for idx, location in enumerate(locations):
    location_data = yearly_summary[yearly_summary["LOCATION"] == location]
    axes[idx].bar(location_data["YEAR"], location_data["WAARDE_SUM"], zorder=10)
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Total Load (kg/year)")
    axes[idx].set_title(f"Yearly Summary for {location}")
    axes[idx].set_ylim(0, yearly_summary["WAARDE_SUM"].max() * 1.1)
    axes[idx].grid(True, alpha=0.3, zorder=-10)

plt.tight_layout()
plt.show()
# %%
import numpy as np

# Create monthly stacked barplot for inflows and outflows with netto flow
if dfs:
    import matplotlib.pyplot as plt

    # Define NAMEs that are leaving (outflows)
    outflows = [
        "Verdamping",
        "Stevin spuisluizen (afsluitdijk)",
        "Lorentz spuisluizen (Afsluitdijk)",
        "lek Spuisluizen (Afsluitdijk)",
        "PWN & WRK III",
        "inlaat Tacozijl (Friesland)",
        "inlaat Teroelsterkolk (Friesland)",
        "inlaat Hooglandgemaal (140)",
        "inlaat Workum",
        "Roggebotsluis",
        "inlaten Noordoostpolder (som)",
        "Krabbersgat spuisluizen (Houtribdijk) afvoer",
        "Houtrib spuisluizen (Houtribdijk) afvoer",
        "Zwarte Water afvoer",
        "Inlaat Vier Noorder Koggen (Westfriesland)",
    ]

    merged_df["YearMonth"] = merged_df["DATUM"].dt.to_period("D")

    # Create a copy and categorize flows
    result_df_copy = merged_df.copy()
    result_df_copy["FlowType"] = result_df_copy["NAME"].apply(
        lambda x: "Outflow" if x in outflows else "Inflow"
    )

    # Group by month and NAME, summing WAARDE
    monthly_flows = (
        result_df_copy.groupby(["YearMonth", "NAME", "FlowType"])["WAARDE"]
        .sum()
        .reset_index()
    )

    # Create pivot table for stacked bar plot
    pivot_inflow = monthly_flows[monthly_flows["FlowType"] == "Inflow"].pivot_table(
        index="YearMonth", columns="NAME", values="WAARDE", aggfunc="sum", fill_value=0
    )
    pivot_outflow = monthly_flows[monthly_flows["FlowType"] == "Outflow"].pivot_table(
        index="YearMonth", columns="NAME", values="WAARDE", aggfunc="sum", fill_value=0
    )

    # Make outflows negative
    pivot_outflow = -pivot_outflow

    # Calculate netto flow (additions - subtractions)
    netto_flow = (pivot_inflow.sum(axis=1) + pivot_outflow.sum(axis=1)).rolling(window=14, center=True, min_periods=1).mean()

    # Process Berging IJsselmeer data
    # berging_ijsselmeer["YearMonth"] = berging_ijsselmeer["DATUM"].dt.to_period("M")
    # berging_monthly = berging_ijsselmeer.groupby("YearMonth")["WAARDE"].sum()

    # Plot stacked bar chart with inflows and negative outflows
    fig, ax = plt.subplots(figsize=(14, 10))

    # Generate distinct colors for each NAME in blue/orange tints
    all_names = list(pivot_inflow.columns) + list(pivot_outflow.columns)
    blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, len(pivot_inflow.columns)))
    orange_shades = plt.cm.Oranges(np.linspace(0.4, 0.9, len(pivot_outflow.columns)))
    color_map = {name: color for name, color in zip(pivot_inflow.columns, blue_shades)}
    color_map.update(
        {name: color for name, color in zip(pivot_outflow.columns, orange_shades)}
    )

    pivot_inflow.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[color_map[col] for col in pivot_inflow.columns],
        alpha=0.8,
        label="Inflows",
    )
    pivot_outflow.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[color_map[col] for col in pivot_outflow.columns],
        alpha=0.8,
        label="Outflows",
    )
    netto_flow.plot(
        kind="line",
        ax=ax,
        marker=None,
        color="black",
        linewidth=1,
        label="Netto Flow",
        zorder=5,
    )

    # ax2 = ax.twinx()
    # berging_monthly.plot(
    #     kind="line",
    #     ax=ax2,
    #     marker=None,
    #     linestyle="--",
    #     color="black",
    #     linewidth=2,
    #     label="Berging IJsselmeer",
    #     markersize=6,
    #     zorder=4,
    # )
    # ax2.set_ylabel("Berging IJsselmeer in miljoen m3", color="green")
    # ax2.tick_params(axis="y", labelcolor="green")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylim(-1_000_000, 1_000_000)
    ax.set_ylabel("Chloride vracht in kg/maand")
    ax.set_title(
        "Monthly Inflows (Additions) and Outflows (Subtractions) with Netto Flow and Berging IJsselmeer"
    )
    ax.set_xlabel("Month")
    ax.legend(title="NAME", bbox_to_anchor=(1.1, 1), loc="upper left")
    ax.grid(axis="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()

# %%

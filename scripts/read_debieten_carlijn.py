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

# %%
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
        fig, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(
            file_data["DATUM"],
            file_data["WAARDE"],
            marker=None,
            linestyle="-",
            color="tab:blue",
            label="Daily",
        )
        ax1.set_xlabel("")
        ax1.set_ylabel("Q in miljoen m3 per dag", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        cumulative = file_data["WAARDE"].cumsum()
        ax2.plot(
            file_data["DATUM"],
            cumulative,
            marker=None,
            linestyle="--",
            color="tab:orange",
            label="Cumulative",
            markersize=3,
        )
        ax2.set_ylabel("Cumulative Q in miljoen m3", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title(f"Q in miljoen m3 per dag - {file}")
        plt.xticks(rotation=45)
        ax1.grid()
        plt.tight_layout()
        plt.show()
# %%

result_df_filt = result_df[result_df["NAME"] != "Berging IJsselmeer"]
berging_ijsselmeer = result_df[result_df["NAME"] == "Berging IJsselmeer"]


result_df_filt["NAME"].unique()
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
    ]  # Add NAME values that are leaving the system

    result_df_filt["YearMonth"] = result_df_filt["DATUM"].dt.to_period("M")

    # Create a copy and categorize flows
    result_df_copy = result_df_filt.copy()
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
    netto_flow = pivot_inflow.sum(axis=1) + pivot_outflow.sum(axis=1)

    # Process Berging IJsselmeer data
    berging_ijsselmeer["YearMonth"] = berging_ijsselmeer["DATUM"].dt.to_period("M")
    berging_monthly = berging_ijsselmeer.groupby("YearMonth")["WAARDE"].sum()

    # Plot stacked bar chart with inflows and negative outflows
    fig, ax = plt.subplots(figsize=(14, 8))

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
        linewidth=1.5,
        label="Netto Flow",
        markersize=8,
        zorder=5,
    )

    ax2 = ax.twinx()
    berging_monthly.plot(
        kind="line",
        ax=ax2,
        marker=None,
        color="green",
        linewidth=2,
        label="Berging IJsselmeer",
        markersize=6,
        zorder=4,
    )
    ax2.set_ylabel("Berging IJsselmeer in miljoen m3", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Q in miljoen m3")
    ax.set_title(
        "Monthly Inflows (Additions) and Outflows (Subtractions) with Netto Flow and Berging IJsselmeer"
    )
    ax.set_xlabel("Month")
    ax.legend(title="NAME", bbox_to_anchor=(1.1, 1), loc="upper left")
    ax.grid(axis="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.show()
# %%
# Stacked barplot showing percentage of inflows and outflows per month
if dfs:
    import matplotlib.pyplot as plt

    result_df_copy = result_df_filt.copy()
    result_df_copy["FlowType"] = result_df_copy["NAME"].apply(
        lambda x: "Outflow" if x in outflows else "Inflow"
    )

    # Filter only inflows
    inflows_df = result_df_copy[result_df_copy["FlowType"] == "Inflow"].copy()
    inflows_df["YearMonth"] = inflows_df["DATUM"].dt.to_period("M")

    # Group by month and NAME, summing WAARDE
    monthly_inflows = (
        inflows_df.groupby(["YearMonth", "NAME"])["WAARDE"].sum().reset_index()
    )

    # Create pivot table
    pivot_inflows = monthly_inflows.pivot_table(
        index="YearMonth", columns="NAME", values="WAARDE", aggfunc="sum", fill_value=0
    )

    # Calculate percentages
    pivot_inflows_pct = pivot_inflows.div(pivot_inflows.sum(axis=1), axis=0) * 100

    # Sort columns by last month percentage (highest to lowest)
    last_month_pct = pivot_inflows_pct.iloc[-1].sort_values(ascending=False)
    pivot_inflows_pct = pivot_inflows_pct[last_month_pct.index]

    # Filter only outflows
    outflow_df = result_df_copy[result_df_copy["FlowType"] == "Outflow"].copy()
    outflow_df["YearMonth"] = outflow_df["DATUM"].dt.to_period("M")

    # Group by month and NAME, summing WAARDE
    monthly_outflows = (
        outflow_df.groupby(["YearMonth", "NAME"])["WAARDE"].sum().reset_index()
    )

    # Create pivot table
    pivot_outflows = monthly_outflows.pivot_table(
        index="YearMonth", columns="NAME", values="WAARDE", aggfunc="sum", fill_value=0
    )

    # Calculate percentages
    pivot_outflows_pct = pivot_outflows.div(pivot_outflows.sum(axis=1), axis=0) * 100

    # Sort columns by last month percentage (highest to lowest)
    last_month_pct = pivot_outflows_pct.iloc[0].sort_values(ascending=False)
    pivot_outflows_pct = pivot_outflows_pct[last_month_pct.index]

    # Plot both stacked percentage bar charts side by side
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    pivot_inflows_pct.plot(kind="bar", stacked=True, ax=ax1, alpha=0.8)
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Monthly Inflow Composition by Source (%)")
    ax1.set_xlabel("Month")
    ax1.legend(
        title="Inflow Source", bbox_to_anchor=(1.05, 1), loc="upper left", reverse=True
    )
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=45)

    pivot_outflows_pct.plot(kind="bar", stacked=True, ax=ax2, alpha=0.8)
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Monthly Outflow Composition by Source (%)")
    ax2.set_xlabel("Month")
    ax2.legend(
        title="Outflow Source", bbox_to_anchor=(1.05, 1), loc="upper left", reverse=True
    )
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
# %%
result_df
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
        file_data["DayOfYear"] = file_data["DATUM"].dt.dayofyear
        file_data["DateNoYear"] = file_data["DATUM"].dt.strftime("%m-%d")

        daily_stats = (
            file_data.groupby("DayOfYear")["WAARDE"].agg(["min", "max"]).reset_index()
        )
        daily_stats["DateNoYear"] = pd.to_datetime(
            daily_stats["DayOfYear"], format="%j"
        ).dt.strftime("%m-%d")

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            daily_stats["DayOfYear"],
            daily_stats["max"],
            marker="o",
            linestyle="-",
            color="tab:blue",
            label="Max",
            markersize=3,
        )
        ax.plot(
            daily_stats["DayOfYear"],
            daily_stats["min"],
            marker="o",
            linestyle="-",
            color="tab:orange",
            label="Min",
            markersize=3,
        )
        ax.fill_between(
            daily_stats["DayOfYear"],
            daily_stats["min"],
            daily_stats["max"],
            alpha=0.2,
            color="tab:gray",
        )

        ax.set_xlabel("Date (MM-DD)")
        ax.set_ylabel("Debiet in miljoen m3 per dag")
        ax.legend()

        plt.title(
            f"Daily Min/Max Debiet in miljoen m3 per dag - periode (2018-2022)\n{file}"
        )
        ax.set_xticks(range(0, 365, 30))
        ax.set_xticklabels(
            [
                pd.to_datetime(str(day + 1), format="%j").strftime("%m-%d")
                for day in range(0, 365, 30)
            ]
        )
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
# %%
# Plot all years on same 365-day axis for comparison
if dfs:
    import matplotlib.pyplot as plt

    files = result_df["NAME"].unique()
    for file in files:
        file_data = (
            result_df[result_df["NAME"] == file]
            .sort_values("DATUM")
            .reset_index(drop=True)
        )
        years = sorted(file_data["DATUM"].dt.year.unique())

        if len(years) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))

            # Add WeekOfYear column
            file_data["WeekOfYear"] = file_data["DATUM"].dt.isocalendar().week

            # Plot each year with different color
            colors = plt.colormaps.get_cmap("tab10")(range(len(years)))
            for i, year in enumerate(years):
                year_data = file_data[file_data["DATUM"].dt.year == year].copy()
                weekly_mean = (
                    year_data.groupby("WeekOfYear")["WAARDE"].mean().reset_index()
                )
                ax.plot(
                    weekly_mean["WeekOfYear"],
                    weekly_mean["WAARDE"],
                    marker="o",
                    linestyle="-",
                    color=colors[i],
                    label=f"{year}",
                    markersize=3,
                    alpha=0.7,
                )

            ax.set_xlabel("Week of Year (1-52)")
            ax.set_ylabel("Debiet in miljoen m3 per dag")
            ax.legend()

            plt.title(f"Daily Debiet Comparison Across Years - {file}")
            ax.set_xticks(range(0, 53, 4))
            ax.set_xticklabels([f"Week {week}" for week in range(0, 53, 4)])
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
# %%

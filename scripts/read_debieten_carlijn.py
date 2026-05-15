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
    result_df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataframe shape: {result_df.shape}")
    print(result_df.head())

    # Optional: Save to CSV
    result_df.to_parquet(
        "P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\3_processed\\debieten_carlijn_2018_2022.parquet",
        index=False,
    )
    print("\nSaved to debieten_carlijn_2018_2022.parquet")
else:
    print("No files were loaded successfully")

# %%
# Find combinations of NAME and FILE
if dfs:
    combinations = result_df[["NAME", "FILE"]].drop_duplicates()
    print(f"\nCombinations of NAME and FILE:\n{combinations}")
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
        "Krabbersgat spuisluizen (Houtribdijk) afvoer",
        "Houtrib spuisluizen (Houtribdijk) afvoer",
        "Zwarte Water afvoer",
        "Inlaat Vier Noorder Koggen (Westfriesland)",
    ]

    result_df_filt["YearMonth"] = result_df_filt["DATUM"].dt.to_period("D")  # M

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

    # Calculate netto flow (additions - subtractions), smoothed with 14-period rolling average
    netto_flow = (
        (pivot_inflow.sum(axis=1) + pivot_outflow.sum(axis=1))
        .rolling(window=14, center=True, min_periods=1)
        .mean()
    )

    # Process Berging IJsselmeer data
    berging_ijsselmeer["YearMonth"] = berging_ijsselmeer["DATUM"].dt.to_period("M")
    berging_monthly = berging_ijsselmeer.groupby("YearMonth")["WAARDE"].sum()

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
        linewidth=0.7,
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
    ax.set_ylabel("Q in miljoen m3")
    ax.set_title(
        "Monthly Inflows (Additions) and Outflows (Subtractions) with Netto Flow and Berging IJsselmeer"
    )
    ax.set_xlabel("Month")
    ax.legend(title="NAME", bbox_to_anchor=(1.1, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(-150, 150)

    # Reduce x-axis tick density
    tick_indices = np.linspace(
        0, len(ax.get_xticklabels()) - 1, min(12, len(ax.get_xticklabels())), dtype=int
    )
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_visible(i in tick_indices)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

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
from io import BytesIO

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

# Yearly mean aggregation
yearly_mean = (
    result_df.groupby(["NAME", result_df["DATUM"].dt.year])["WAARDE"]
    .mean()
    .reset_index()
)
yearly_mean.columns = ["NAME", "year", "WAARDE"]

# Pre-group for fast lookup (avoids filtering in loop)
yearly_dict: dict[str, np.ndarray] = {}
for name, group in yearly_mean.groupby("NAME"):
    yearly_dict[name] = group.sort_values("year")

locaties = gpd.read_file(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\3_processed\locaties_debiet.gpkg"
)
locaties
# %%
# Reproject to Web Mercator
locaties_3857 = locaties.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 10))

# Plot first (sets extent correctly)
# Determine marker and color based on Invoer/Afvoer columns
colors = []
markers = []
sizes = []
for idx, row in locaties_3857.iterrows():
    if row.get("Invoer") == "Ja":
        colors.append("blue")
        markers.append("^")
    elif row.get("Afvoer") == "Ja":
        colors.append("orange")
        markers.append("v")
    else:
        colors.append("red")
        markers.append("o")

    # Get marker size based on last year's mean value
    name = row["naam"]
    if name in yearly_dict and len(yearly_dict[name]) > 0:
        last_year_mean = yearly_dict[name].iloc[-1]["WAARDE"]
        # Normalize size: scale last year mean to reasonable marker size
        markersize = max(50, min(500, last_year_mean * 30))
    else:
        markersize = 200
    sizes.append(markersize)

for i, (idx, row) in enumerate(locaties_3857.iterrows()):
    locaties_3857.iloc[i : i + 1].plot(
        ax=ax,
        alpha=0.5,
        markersize=sizes[i],
        color=colors[i],
        marker=markers[i],
        zorder=2,
    )
    # Add location name as text
    ax.text(
        row.geometry.x + 2500,
        row.geometry.y + 2500,
        row["naam"],
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
        zorder=3,
    )

# Then add basemap
ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.Positron,
    zoom=11,
)

ax.set_title("Debiet Locations")

plt.tight_layout()
plt.show()
# %%
import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np

# --- Prepare data -------------------------------------------------------------

# Ensure CRS is Web Mercator
locaties_3857 = locaties.to_crs(epsg=3857)
locaties_3857
# %%
# --- Main figure --------------------------------------------------------------

fig, ax = plt.subplots(figsize=(18, 16))

# 1. Plot first → sets correct extent
locaties_3857.plot(
    ax=ax,
    alpha=0.8,
    edgecolor="k",
    markersize=40,
    color="black",
    zorder=2,
)

ax.set_title("Yearly Mean Debiet by Location with Barplots")

# Get current limits and expand
xlim = ax.get_xlim()
ylim = ax.get_ylim()
x_range = xlim[1] - xlim[0]
y_range = ylim[1] - ylim[0]
ax.set_xlim(xlim[0] - 0.1 * x_range, xlim[1] + 0.2 * x_range)
ax.set_ylim(ylim[0] - 0.1 * y_range, ylim[1] + 0.1 * y_range)

# 2. Then basemap
ctx.add_basemap(
    ax,
    source=ctx.providers.CartoDB.Positron,
    zoom=11,
)

# Web mercator → axis labels not meaningful
ax.set_axis_off()

# --- Add inset barplots -------------------------------------------------------

# Manual plot location overrides (location_name: (x, y))
plot_location_overrides = {
    "inlaat Teroelsterkolk (Friesland)": (623683.103, 6964401.631),
    "inlaat Tacozijl (Friesland)": (624219.029, 6947166.574),
    "gemaal Buma (Noordoostpolder)": (642031.175, 6962973.111),
    "gemaal Wouda (Friesland)": (642294.701, 6944600.491),
    "PWN & WRK III": (583925.035, 6940385.894),
    "gemaal Colijn laag (Fleverwaard)": (630914, 6890000),
    "gemaal Colijn hoog (Fleverwaard)": (630886, 6905541.395),
    "IJssel": (663080.805, 6902965.655),
    "Roggebotsluis": (648500, 6890000),
    "Gemaal Roggebot": (665600, 6890000),
    "Roggebotspui": (683500, 6890000),
    "Zwarte Water": (674808, 6916009),
    "gemaal Stroink (Wold en Wieden)": (664933.321, 6938447.449),
    "gemaal Smeenge (Noordoostpolder)": (652150.545, 6921989.029),
    "Houtrib spuisluizen (Houtribdijk) aanvoer": (602942, 6893942),
    "Houtrib spuisluizen (Houtribdijk) afvoer": (604942, 6908942),
    "Krabbersgat spuisluizen (Houtribdijk) aanvoer": (585987, 6912200),
    "Krabbersgat spuisluizen (Houtribdijk) afvoer": (587987, 6926200),
    "Zwarte Water aanvoer": (671208, 6918024),
    "Zwarte Water afvoer": (689708, 6918024),
}

# Calculate global y-axis limits
y_min = min(
    [
        yearly_dict[loc.naam]["WAARDE"].min()
        for loc in locaties_3857.itertuples()
        if loc.naam in yearly_dict
    ]
)
y_max = max(
    [
        yearly_dict[loc.naam]["WAARDE"].max()
        for loc in locaties_3857.itertuples()
        if loc.naam in yearly_dict
    ]
)

for loc in locaties_3857.itertuples():
    location_name: str = loc.naam

    if location_name not in yearly_dict:
        continue

    location_data = yearly_dict[location_name]

    # Determine color based on Invoer/Afvoer columns
    if loc.Invoer == "Ja":
        bar_color = "blue"
    elif loc.Afvoer == "Ja":
        bar_color = "orange"
    else:
        bar_color = "red"

    # --- Create small barplot (lightweight) -----------------------------------
    fig_bar, ax_bar = plt.subplots(figsize=(2.3, 1.7), dpi=100)

    ax_bar.bar(
        location_data["year"].astype(str),
        location_data["WAARDE"].values,
        color=bar_color,
        zorder=2,
    )

    ax_bar.set_xticks(range(len(location_data["year"])))
    ax_bar.set_xticklabels(
        location_data["year"].astype(str),
        fontsize=9,
    )

    # ax_bar.set_ylim(y_min, y_max)
    # ax_bar.set_ylabel('Mean Debiet', fontsize=8)
    ax_bar.grid(axis="y", alpha=0.3, zorder=1)
    # ax_bar.set_yticks([])
    ax_bar.set_title(location_name, fontsize=10)

    # Tight layout inside small figure
    fig_bar.tight_layout(pad=0.5)

    # --- Convert to image -----------------------------------------------------
    buf = BytesIO()
    fig_bar.savefig(buf, format="png")
    buf.seek(0)

    img_array = np.array(Image.open(buf))

    # --- Place on map ---------------------------------------------------------
    # Use override location if specified, otherwise use original location
    if location_name in plot_location_overrides:
        plot_xy = plot_location_overrides[location_name]
        original_xy = (loc.geometry.x, loc.geometry.y)

        # Draw line from plot to original location
        ax.plot(
            [plot_xy[0], original_xy[0]],
            [plot_xy[1], original_xy[1]],
            color="black",
            linestyle="-",
            linewidth=2,
            alpha=0.9,
            zorder=1,
        )
    else:
        plot_xy = (loc.geometry.x, loc.geometry.y)

    imagebox = OffsetImage(img_array, zoom=0.5)

    ab = AnnotationBbox(
        imagebox,
        plot_xy,
        frameon=True,
        pad=0.2,
        zorder=3,
    )

    ax.add_artist(ab)

    plt.close(fig_bar)

# --- Final layout -------------------------------------------------------------

# plt.tight_layout()
plt.show()
# %%

# %%
from pathlib import Path
import datetime as dt
TODAY = dt.datetime.today().strftime("%Y%m%d")

import pandas as pd

# Directory to search for .YMR files
debiet_dir = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\data_vanCarlijn\Qnet.2018_2022\Qnet"

# Find all .YMR files
ymr_files = list(Path(debiet_dir).glob("*.YMR"))
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
                # Convert from million m3/d to m3/s: (million m3/d) * 1e6 / 86400
                if first_line != "Berging IJsselmeer":
                    df["WAARDE"] = df["WAARDE"] * 1e6 / 86400
                    df["UNIT"] = "m³/s"
                else:
                    df["UNIT"] = "million m³"
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

    # Print min and max years
    min_year = result_df["DATUM"].dt.year.min()
    max_year = result_df["DATUM"].dt.year.max()
    print(f"\nData year range: {min_year} - {max_year}")

    # Optional: Save to Parquet
    result_df.to_parquet(
        f"P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\3_processed\\debieten_carlijn_YMR_{min_year}_{max_year}.parquet",
        index=False,
    )
    print(f"\nSaved to debieten_carlijn_YMR_{min_year}_{max_year}.parquet")
else:
    print("No files were loaded successfully")
result_df
# %%
# Find combinations of NAME and FILE
if dfs:
    combinations = result_df[["NAME", "FILE"]].drop_duplicates()
    print(f"\nCombinations of NAME and FILE:\n{combinations}")
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
        ax1.set_ylabel("Q in miljoen m³ per dag", color="tab:blue")
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
        ax2.set_ylabel("Cumulative Q in miljoen m³", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        plt.title(f"Q in miljoen m³ per dag - {file}")
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
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    # Define NAMEs that are leaving (outflows)
    outflows = [
        "Verdamping",
        "Stevin spuisluizen (Afsluitdijk)",
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

    result_df_filt["Day"] = result_df_filt["DATUM"].dt.floor("D")

    # Create a copy and categorize flows
    result_df_copy = result_df_filt.copy()
    result_df_copy["FlowType"] = result_df_copy["NAME"].apply(
        lambda x: "Outflow" if x in outflows else "Inflow"
    )

    # Group by day and NAME, summing WAARDE
    daily_flows = (
        result_df_copy.groupby(["Day", "NAME", "FlowType", "UNIT"])["WAARDE"]
        .sum()
        .reset_index()
    )

    # Complete daily index so inflow, outflow, and netto_flow share identical daily spacing
    all_days = pd.date_range(
        daily_flows["Day"].min(), daily_flows["Day"].max(), freq="D"
    )

    # Create pivot tables aligned to the complete daily index
    pivot_inflow = (
        daily_flows[daily_flows["FlowType"] == "Inflow"]
        .pivot_table(
            index="Day", columns="NAME", values="WAARDE", aggfunc="sum", fill_value=0
        )
        .reindex(all_days, fill_value=0)
    )
    pivot_outflow = (
        daily_flows[daily_flows["FlowType"] == "Outflow"]
        .pivot_table(
            index="Day", columns="NAME", values="WAARDE", aggfunc="sum", fill_value=0
        )
        .reindex(all_days, fill_value=0)
    )

    # Make outflows negative
    pivot_outflow = -pivot_outflow

    # Sort inflows and outflows by total magnitude descending (largest closest to 0 and first in legend)
    pivot_inflow = pivot_inflow[
        pivot_inflow.abs().sum().sort_values(ascending=False).index
    ]
    pivot_outflow = pivot_outflow[
        pivot_outflow.abs().sum().sort_values(ascending=False).index
    ]

    # Calculate netto flow (additions - subtractions), smoothed with 14-day rolling average
    netto_flow = (
        (pivot_inflow.sum(axis=1) + pivot_outflow.sum(axis=1))
        .rolling(window=14, center=True, min_periods=1)
        .mean()
    )

    # Process Berging IJsselmeer data
    berging_ijsselmeer["Day"] = berging_ijsselmeer["DATUM"].dt.floor("D")
    berging_daily = (
        berging_ijsselmeer.groupby("Day")["WAARDE"].sum().reindex(all_days)
    )

    # Plot stacked bar chart with inflows and negative outflows on the same datetime axis
    fig, ax = plt.subplots(figsize=(14, 10))

    # Generate distinct colors for each NAME in blue/orange tints
    blue_shades = plt.cm.Blues(np.linspace(0.4, 0.9, len(pivot_inflow.columns)))
    orange_shades = plt.cm.Oranges(np.linspace(0.4, 0.9, len(pivot_outflow.columns)))
    color_map = {name: color for name, color in zip(pivot_inflow.columns, blue_shades)}
    color_map.update(
        {name: color for name, color in zip(pivot_outflow.columns, orange_shades)}
    )

    bottom_in = np.zeros(len(all_days))
    for col in pivot_inflow.columns:
        vals = pivot_inflow[col].values
        ax.bar(
            all_days,
            vals,
            bottom=bottom_in,
            width=1.0,
            color=color_map[col],
            alpha=0.8,
            label=col,
        )
        bottom_in += vals

    bottom_out = np.zeros(len(all_days))
    for col in pivot_outflow.columns:
        vals = pivot_outflow[col].values
        ax.bar(
            all_days,
            vals,
            bottom=bottom_out,
            width=1.0,
            color=color_map[col],
            alpha=0.8,
            label=col,
        )
        bottom_out += vals

    ax.plot(
        all_days,
        netto_flow.values,
        marker=None,
        color="black",
        linewidth=0.7,
        label="Netto Flow",
        zorder=5,
    )

    # ax2 = ax.twinx()
    # ax2.plot(
    #     all_days,
    #     berging_daily.values,
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

    unit = ", ".join(daily_flows["UNIT"].dropna().unique())

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(f"Q ({unit})")
    ax.set_title("Dagelijkse gesommeerde In- en Uitstroom debiet en netto debiet IJsselmeer")
    # ax.set_xlabel("Dag")
    ax.legend(title="In- en uitstroom locaties", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.grid(axis="x", linestyle="--", linewidth=0.25)    
    ax.set_ylim(-1750, 1750)

    # Format daily datetime x-axis cleanly
    ax.set_xlim(all_days.min()-pd.Timedelta(days=14), all_days.max()+pd.Timedelta(days=14))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=15, maxticks=30))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    plt.setp(ax.get_xticklabels(), ha="center")

    plt.tight_layout()
    plt.savefig(
        rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\debiet\daily_in_out_netto_flow_YMR_{TODAY}.png",
        dpi=300,
        bbox_inches="tight",
    )
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

    inflow_months = pivot_inflows_pct.index.to_timestamp()
    outflow_months = pivot_outflows_pct.index.to_timestamp()

    # Plot both stacked percentage bar charts side by side
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))

    bottom_in = np.zeros(len(inflow_months))
    for col in pivot_inflows_pct.columns:
        vals = pivot_inflows_pct[col].values
        ax1.bar(inflow_months, vals, bottom=bottom_in, width=20, alpha=0.8, label=col)
        bottom_in += vals
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Maandelijkse Instroom debiet per locatie (%)")
    ax1.set_xlabel("")
    ax1.legend(
        title="Instroom locatie", bbox_to_anchor=(1.01, 1), loc="upper left", reverse=True
    )
    ax1.set_ylim(0, 100)
    # Format monthly datetime x-axis cleanly
    ax1.set_xlim(
        inflow_months.min() - pd.Timedelta(days=15),
        inflow_months.max() + pd.Timedelta(days=15),
    )
    inflow_interval = max(1, len(inflow_months) // 24)
    ax1.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=range(1, 13, inflow_interval))
    )
    ax1.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            ax1.xaxis.get_major_locator(),
            zero_formats=["", "%b\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
            show_offset=False,
        )
    )
    plt.setp(ax1.get_xticklabels(), ha="center")

    bottom_out = np.zeros(len(outflow_months))
    for col in pivot_outflows_pct.columns:
        vals = pivot_outflows_pct[col].values
        ax2.bar(outflow_months, vals, bottom=bottom_out, width=20, alpha=0.8, label=col)
        bottom_out += vals
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Maandelijkse Uitstroom debiet per locatie (%)")
    ax2.set_xlabel("")
    ax2.legend(
        title="Uitstroom locatie", bbox_to_anchor=(1.01, 1), loc="upper left", reverse=True
    )
    ax2.set_ylim(0, 100)
    # Format monthly datetime x-axis cleanly
    ax2.set_xlim(
        outflow_months.min() - pd.Timedelta(days=15),
        outflow_months.max() + pd.Timedelta(days=15),
    )
    outflow_interval = max(1, len(outflow_months) // 24)
    ax2.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=range(1, 13, outflow_interval))
    )
    ax2.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            ax2.xaxis.get_major_locator(),
            zero_formats=["", "%b\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
            show_offset=False,
        )
    )
    plt.setp(ax2.get_xticklabels(), ha="center")

    plt.tight_layout()
    plt.savefig(
        rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\debiet\monthly_in_out_percentage_YMR_{TODAY}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# %%
# Plot daily max and min WAARDE per location (by day of year)
# if dfs:
#     import matplotlib.pyplot as plt

#     files = result_df["NAME"].unique()
#     for file in files:
#         file_data = (
#             result_df[result_df["NAME"] == file]
#             .sort_values("DATUM")
#             .reset_index(drop=True)
#         )
#         file_data["DayOfYear"] = file_data["DATUM"].dt.dayofyear
#         file_data["DateNoYear"] = file_data["DATUM"].dt.strftime("%m-%d")

#         daily_stats = (
#             file_data.groupby("DayOfYear")["WAARDE"].agg(["min", "max"]).reset_index()
#         )
#         daily_stats["DateNoYear"] = pd.to_datetime(
#             daily_stats["DayOfYear"], format="%j"
#         ).dt.strftime("%m-%d")

#         fig, ax = plt.subplots(figsize=(14, 6))

#         ax.plot(
#             daily_stats["DayOfYear"],
#             daily_stats["max"],
#             marker="o",
#             linestyle="-",
#             color="tab:blue",
#             label="Max",
#             markersize=3,
#         )
#         ax.plot(
#             daily_stats["DayOfYear"],
#             daily_stats["min"],
#             marker="o",
#             linestyle="-",
#             color="tab:orange",
#             label="Min",
#             markersize=3,
#         )
#         ax.fill_between(
#             daily_stats["DayOfYear"],
#             daily_stats["min"],
#             daily_stats["max"],
#             alpha=0.2,
#             color="tab:gray",
#         )

#         ax.set_xlabel("Date (MM-DD)")
#         ax.set_ylabel("Debiet in miljoen m³ per dag")
#         ax.legend()

#         plt.title(
#             f"Daily Min/Max Debiet in miljoen m³ per dag - periode (2018-2022)\n{file}"
#         )
#         ax.set_xticks(range(0, 365, 30))
#         ax.set_xticklabels(
#             [
#                 pd.to_datetime(str(day + 1), format="%j").strftime("%m-%d")
#                 for day in range(0, 365, 30)
#             ]
#         )
#         ax.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.show()
# %%
# Plot all years on same 365-day axis for comparison
# if dfs:
#     import matplotlib.pyplot as plt

#     files = result_df["NAME"].unique()
#     for file in files:
#         file_data = (
#             result_df[result_df["NAME"] == file]
#             .sort_values("DATUM")
#             .reset_index(drop=True)
#         )
#         years = sorted(file_data["DATUM"].dt.year.unique())

#         if len(years) > 0:
#             fig, ax = plt.subplots(figsize=(14, 6))

#             # Add WeekOfYear column
#             file_data["WeekOfYear"] = file_data["DATUM"].dt.isocalendar().week

#             # Plot each year with different color
#             colors = plt.colormaps.get_cmap("tab10")(range(len(years)))
#             for i, year in enumerate(years):
#                 year_data = file_data[file_data["DATUM"].dt.year == year].copy()
#                 weekly_mean = (
#                     year_data.groupby("WeekOfYear")["WAARDE"].mean().reset_index()
#                 )
#                 ax.plot(
#                     weekly_mean["WeekOfYear"],
#                     weekly_mean["WAARDE"],
#                     marker="o",
#                     linestyle="-",
#                     color=colors[i],
#                     label=f"{year}",
#                     markersize=3,
#                     alpha=0.7,
#                 )

#             ax.set_xlabel("Week of Year (1-52)")
#             ax.set_ylabel("Debiet in miljoen m³ per dag")
#             ax.legend()

#             plt.title(f"Daily Debiet Comparison Across Years - {file}")
#             ax.set_xticks(range(0, 53, 4))
#             ax.set_xticklabels([f"Week {week}" for week in range(0, 53, 4)])
#             ax.grid(alpha=0.3)
#             plt.tight_layout()
#             plt.show()
# %%
# from io import BytesIO
# from pathlib import Path

# import contextily as ctx
# import geopandas as gpd
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.offsetbox import AnnotationBbox, OffsetImage
# from PIL import Image

# DATA_GIS_DIR = Path(
#     r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\GIS"
# )

# # Yearly mean aggregation
# yearly_mean = (
#     result_df.groupby(["NAME", result_df["DATUM"].dt.year])["WAARDE"]
#     .mean()
#     .reset_index()
# )
# yearly_mean.columns = ["NAME", "year", "WAARDE"]

# # Pre-group for fast lookup (avoids filtering in loop)
# yearly_dict: dict[str, np.ndarray] = {}
# for name, group in yearly_mean.groupby("NAME"):
#     yearly_dict[name] = group.sort_values("year")

# locaties = gpd.read_file(DATA_GIS_DIR / "locaties_debiet.gpkg")
# locaties
# # %%
# # Reproject to Web Mercator
# locaties_3857 = locaties.to_crs(epsg=3857)

# fig, ax = plt.subplots(figsize=(12, 10))

# # Plot first (sets extent correctly)
# # Determine marker and color based on Invoer/Afvoer columns
# colors = []
# markers = []
# sizes = []
# for idx, row in locaties_3857.iterrows():
#     if row.get("Invoer") == "Ja":
#         colors.append("blue")
#         markers.append("^")
#     elif row.get("Afvoer") == "Ja":
#         colors.append("orange")
#         markers.append("v")
#     else:
#         colors.append("red")
#         markers.append("o")

#     # Get marker size based on last year's mean value
#     name = row["naam"]
#     if name in yearly_dict and len(yearly_dict[name]) > 0:
#         last_year_mean = yearly_dict[name].iloc[-1]["WAARDE"]
#         # Normalize size: scale last year mean to reasonable marker size
#         markersize = max(50, min(500, last_year_mean * 30))
#     else:
#         markersize = 200
#     sizes.append(markersize)

# for i, (idx, row) in enumerate(locaties_3857.iterrows()):
#     locaties_3857.iloc[i : i + 1].plot(
#         ax=ax,
#         alpha=0.5,
#         markersize=sizes[i],
#         color=colors[i],
#         marker=markers[i],
#         zorder=2,
#     )
#     # Add location name as text
#     ax.text(
#         row.geometry.x + 2500,
#         row.geometry.y + 2500,
#         row["naam"],
#         fontsize=9,
#         ha="center",
#         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
#         zorder=3,
#     )

# # Then add basemap
# ctx.add_basemap(
#     ax,
#     source=ctx.providers.CartoDB.Positron,
#     zoom=11,
# )

# ax.set_title("Debiet Locations")
# TODAY = pd.Timestamp.now().strftime("%Y%m%d")
# plt.tight_layout()
# # plt.savefig(
# #     rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\debiet\debiet_locations_map_20260720.png",
# #     dpi=300,
# # )
# plt.show()
# %%
# import contextily as ctx
# import matplotlib.pyplot as plt
# import numpy as np

# # --- Prepare data -------------------------------------------------------------

# # Ensure CRS is Web Mercator
# locaties_3857 = locaties.to_crs(epsg=3857)
# locaties_3857

# # --- Main figure --------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(18, 16))

# # 1. Plot first → sets correct extent
# locaties_3857.plot(
#     ax=ax,
#     alpha=0.8,
#     edgecolor="k",
#     markersize=40,
#     color="black",
#     zorder=2,
# )

# ax.set_title("Yearly Mean Debiet by Location with Barplots")

# # Get current limits and expand
# xlim = ax.get_xlim()
# ylim = ax.get_ylim()
# x_range = xlim[1] - xlim[0]
# y_range = ylim[1] - ylim[0]
# ax.set_xlim(xlim[0] - 0.1 * x_range, xlim[1] + 0.2 * x_range)
# ax.set_ylim(ylim[0] - 0.1 * y_range, ylim[1] + 0.1 * y_range)

# # 2. Then basemap
# ctx.add_basemap(
#     ax,
#     source=ctx.providers.CartoDB.Positron,
#     zoom=11,
# )

# # Web mercator → axis labels not meaningful
# ax.set_axis_off()

# # --- Add inset barplots -------------------------------------------------------

# # Manual plot location overrides (location_name: (x, y))
# plot_location_overrides = {
#     "inlaat Teroelsterkolk (Friesland)": (623683.103, 6964401.631),
#     "inlaat Tacozijl (Friesland)": (624219.029, 6947166.574),
#     "gemaal Buma (Noordoostpolder)": (642031.175, 6962973.111),
#     "gemaal Wouda (Friesland)": (642294.701, 6944600.491),
#     "PWN & WRK III": (583925.035, 6940385.894),
#     "gemaal Colijn laag (Fleverwaard)": (630914, 6890000),
#     "gemaal Colijn hoog (Fleverwaard)": (630886, 6905541.395),
#     "IJssel": (663080.805, 6902965.655),
#     "Roggebotsluis": (648500, 6890000),
#     "Gemaal Roggebot": (665600, 6890000),
#     "Roggebotspui": (683500, 6890000),
#     "Zwarte Water": (674808, 6916009),
#     "gemaal Stroink (Wold en Wieden)": (664933.321, 6938447.449),
#     "gemaal Smeenge (Noordoostpolder)": (652150.545, 6921989.029),
#     "Houtrib spuisluizen (Houtribdijk) aanvoer": (602942, 6893942),
#     "Houtrib spuisluizen (Houtribdijk) afvoer": (604942, 6908942),
#     "Krabbersgat spuisluizen (Houtribdijk) aanvoer": (585987, 6912200),
#     "Krabbersgat spuisluizen (Houtribdijk) afvoer": (587987, 6926200),
#     "Zwarte Water aanvoer": (671208, 6918024),
#     "Zwarte Water afvoer": (689708, 6918024),
# }

# # Calculate global y-axis limits
# y_min = min(
#     [
#         yearly_dict[loc.naam]["WAARDE"].min()
#         for loc in locaties_3857.itertuples()
#         if loc.naam in yearly_dict
#     ]
# )
# y_max = max(
#     [
#         yearly_dict[loc.naam]["WAARDE"].max()
#         for loc in locaties_3857.itertuples()
#         if loc.naam in yearly_dict
#     ]
# )

# for loc in locaties_3857.itertuples():
#     location_name: str = loc.naam

#     if location_name not in yearly_dict:
#         continue

#     location_data = yearly_dict[location_name]

#     # Determine color based on Invoer/Afvoer columns
#     if loc.Invoer == "Ja":
#         bar_color = "blue"
#     elif loc.Afvoer == "Ja":
#         bar_color = "orange"
#     else:
#         bar_color = "red"

#     # --- Create small barplot (lightweight) -----------------------------------
#     fig_bar, ax_bar = plt.subplots(figsize=(2.3, 1.7), dpi=100)

#     ax_bar.bar(
#         location_data["year"].astype(str),
#         location_data["WAARDE"].values,
#         color=bar_color,
#         zorder=2,
#     )

#     ax_bar.set_xticks(range(len(location_data["year"])))
#     ax_bar.set_xticklabels(
#         location_data["year"].astype(str),
#         fontsize=9,
#     )

#     # ax_bar.set_ylim(y_min, y_max)
#     # ax_bar.set_ylabel('Mean Debiet', fontsize=8)
#     ax_bar.grid(axis="y", alpha=0.3, zorder=1)
#     # ax_bar.set_yticks([])
#     ax_bar.set_title(location_name, fontsize=10)

#     # Tight layout inside small figure
#     fig_bar.tight_layout(pad=0.5)

#     # --- Convert to image -----------------------------------------------------
#     buf = BytesIO()
#     fig_bar.savefig(buf, format="png")
#     buf.seek(0)

#     img_array = np.array(Image.open(buf))

#     # --- Place on map ---------------------------------------------------------
#     # Use override location if specified, otherwise use original location
#     if location_name in plot_location_overrides:
#         plot_xy = plot_location_overrides[location_name]
#         original_xy = (loc.geometry.x, loc.geometry.y)

#         # Draw line from plot to original location
#         ax.plot(
#             [plot_xy[0], original_xy[0]],
#             [plot_xy[1], original_xy[1]],
#             color="black",
#             linestyle="-",
#             linewidth=2,
#             alpha=0.9,
#             zorder=1,
#         )
#     else:
#         plot_xy = (loc.geometry.x, loc.geometry.y)

#     imagebox = OffsetImage(img_array, zoom=0.5)

#     ab = AnnotationBbox(
#         imagebox,
#         plot_xy,
#         frameon=True,
#         pad=0.2,
#         zorder=3,
#     )

#     ax.add_artist(ab)

#     plt.close(fig_bar)

# # --- Final layout -------------------------------------------------------------

# # plt.tight_layout()
# plt.show()
# %%
import numpy as np
import plotly.graph_objects as go

# Create daily interactive plot for inflows and outflows with netto flow
if dfs:
    # Define NAMEs that are leaving (outflows)
    outflows = [
        "Verdamping",
        "Stevin spuisluizen (Afsluitdijk)",
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

    result_df_filt["Year"] = result_df_filt["DATUM"].dt.year
    result_df_filt["Date"] = result_df_filt["DATUM"].dt.date

    # Create a copy and categorize flows
    result_df_copy = result_df_filt.copy()

    # Define IJssel inflow and Stevin/Lorentz outflows specifically
    ijssel_inflow = ["IJssel"]
    stevin_outflows = ["Stevin spuisluizen (Afsluitdijk)"]
    lorentz_outflows = ["Lorentz spuisluizen (Afsluitdijk)"]

    def categorize_flow(name):
        if name in ijssel_inflow:
            return "IJssel"
        elif name in stevin_outflows:
            return "Stevinsluizen"
        elif name in lorentz_outflows:
            return "Lorentzsluizen"
        elif name in outflows:
            return "Overig uit"
        else:
            return "Overig in"

    result_df_copy["FlowType"] = result_df_copy["NAME"].apply(categorize_flow)

    # Group by year, date and NAME, summing WAARDE
    daily_flows = (
        result_df_copy.groupby(["Year", "DATUM", "NAME", "FlowType"])["WAARDE"]
        .sum()
        .reset_index()
    )

    # Get unique years for dropdown
    years = sorted(daily_flows["Year"].unique())

    # Define colors
    colors = {
        "IJssel": "darkblue",
        "Overig in": "lightblue",
        "Stevinsluizen": "darkred",
        "Lorentzsluizen": "red",
        "Overig uit": "lightsalmon",
    }

    # Create buttons for year dropdown
    buttons = []
    for year in years:
        year_data = daily_flows[daily_flows["Year"] == year].sort_values("DATUM")

        # Get date values for x-axis
        dates = sorted(year_data["DATUM"].unique())

        # Create pivot tables for each category
        pivot_ijssel = (
            year_data[year_data["FlowType"] == "IJssel"]
            .groupby("DATUM")["WAARDE"]
            .sum()
        )
        pivot_other_inflow = (
            year_data[year_data["FlowType"] == "Overig in"]
            .groupby("DATUM")["WAARDE"]
            .sum()
        )
        pivot_stevin = (
            -year_data[year_data["FlowType"] == "Stevinsluizen"]
            .groupby("DATUM")["WAARDE"]
            .sum()
        )
        pivot_lorentz = (
            -year_data[year_data["FlowType"] == "Lorentzsluizen"]
            .groupby("DATUM")["WAARDE"]
            .sum()
        )
        pivot_other_outflow = (
            -year_data[year_data["FlowType"] == "Overig uit"]
            .groupby("DATUM")["WAARDE"]
            .sum()
        )

        # Calculate netto flow
        netto_flow = (
            pivot_ijssel
            + pivot_other_inflow
            + pivot_stevin
            + pivot_lorentz
            + pivot_other_outflow
        )

        buttons.append(
            dict(
                label=str(year),
                method="update",
                args=[
                    {
                        "x": [dates] * 6,
                        "y": [
                            [pivot_ijssel.get(d, 0) for d in dates],
                            [pivot_other_inflow.get(d, 0) for d in dates],
                            [pivot_stevin.get(d, 0) for d in dates],
                            [pivot_lorentz.get(d, 0) for d in dates],
                            [pivot_other_outflow.get(d, 0) for d in dates],
                            [netto_flow.get(d, 0) for d in dates],
                        ],
                    },
                    {"title": f"Daily Inflows and Outflows - {year}"},
                ],
            )
        )

    # Create fresh figure with first year's data
    first_year = years[0]
    first_year_data = daily_flows[daily_flows["Year"] == first_year].sort_values(
        "DATUM"
    )

    dates = sorted(first_year_data["DATUM"].unique())

    pivot_ijssel = (
        first_year_data[first_year_data["FlowType"] == "IJssel"]
        .groupby("DATUM")["WAARDE"]
        .sum()
    )
    pivot_other_inflow = (
        first_year_data[first_year_data["FlowType"] == "Overig in"]
        .groupby("DATUM")["WAARDE"]
        .sum()
    )
    pivot_stevin = (
        -first_year_data[first_year_data["FlowType"] == "Stevinsluizen"]
        .groupby("DATUM")["WAARDE"]
        .sum()
    )
    pivot_lorentz = (
        -first_year_data[first_year_data["FlowType"] == "Lorentzsluizen"]
        .groupby("DATUM")["WAARDE"]
        .sum()
    )
    pivot_other_outflow = (
        -first_year_data[first_year_data["FlowType"] == "Overig uit"]
        .groupby("DATUM")["WAARDE"]
        .sum()
    )

    netto_flow = (
        pivot_ijssel
        + pivot_other_inflow
        + pivot_stevin
        + pivot_lorentz
        + pivot_other_outflow
    )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_ijssel.get(d, 0) for d in dates],
            name="IJssel",
            marker=dict(color=colors["IJssel"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_other_inflow.get(d, 0) for d in dates],
            name="Overig in",
            marker=dict(color=colors["Overig in"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_stevin.get(d, 0) for d in dates],
            name="Stevinsluizen",
            marker=dict(color=colors["Stevinsluizen"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_lorentz.get(d, 0) for d in dates],
            name="Lorentzsluizen",
            marker=dict(color=colors["Lorentzsluizen"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_other_outflow.get(d, 0) for d in dates],
            name="Overig uit",
            marker=dict(color=colors["Overig uit"]),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[netto_flow.get(d, 0) for d in dates],
            name="Netto Flow",
            mode="lines",
            line=dict(color="black", width=1.5),
            yaxis="y",
        )
    )

    fig.update_layout(
        barmode="relative",
        title=f"Daily Inflows and Outflows - {first_year}",
        # xaxis_title="Date",
        yaxis_title="Q in miljoen m³",
        hovermode="x unified",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ],
        height=600,
        width=1200,
    )

    # fig.write_html("interactive_flows.html")
    fig.show()

# %%
result_df["NAME"].unique()
# %%
# Plot WAARDE per DATUM for each FILE
if dfs:
    import matplotlib.pyplot as plt

    # 1) List of locations to include (empty = all)
    locations_to_include = [
        "IJssel",
        "Neerslag",
        "Verdamping",
        "Houtrib spuisluizen (Houtribdijk) aanvoer",
        "Houtrib spuisluizen (Houtribdijk) afvoer",
        "Krabbersgat spuisluizen (Houtribdijk) aanvoer",
        "Krabbersgat spuisluizen (Houtribdijk) afvoer",
        "Stevin spuisluizen (Afsluitdijk)",
        "Lorentz spuisluizen (Afsluitdijk)",
        "Berging IJsselmeer",
    ]

    # Define NAMEs that are leaving (outflows)
    outflows = [
        "Verdamping",
        "Stevin spuisluizen (Afsluitdijk)",
        "Lorentz spuisluizen (Afsluitdijk)",
        "inlaten Noordoostpolder (som)",
        "Krabbersgat spuisluizen (Houtribdijk) afvoer",
        "Houtrib spuisluizen (Houtribdijk) afvoer",
        "lek Spuisluizen (Afsluitdijk)",
        "PWN & WRK III",
        "inlaat Tacozijl (Friesland)",
        "inlaat Teroelsterkolk (Friesland)",
        "inlaat Hooglandgemaal (140)",
        "inlaat Workum",
        "Roggebotsluis",
        "inlaten Noordoostpolder (som)",
        "Zwarte Water afvoer",
        "Inlaat Vier Noorder Koggen (Westfriesland)",
    ]

    output_dir = Path(
        rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define flow categories for colored spans (only relevant for IJssel)
    flow_categories = [
        # {"ymin": 1150, "ymax": float("inf"), "color": "#E59621", "va": "bottom", "alpha": 0.25, "label": "Hoge afvoer (>1150)"},
        {
            "ymin": 740,
            "ymax": 1150,
            "color": "#DCC814",
            "va": "bottom",
            "alpha": 0.45,
            "label": "Verhoogde afvoer",
        },
        {
            "ymin": 640,
            "ymax": 740,
            "color": "#B1FF3B",
            "va": "bottom",
            "alpha": 0.45,
            "label": "Licht verhoogd",
        },
        {
            "ymin": 165,
            "ymax": 640,
            "color": "#1B7F1B",
            "va": "bottom",
            "alpha": 0.45,
            "label": "Normale afvoer",
        },
        {
            "ymin": 0,
            "ymax": 165,
            "color": "#966921",
            "va": "top",
            "alpha": 0.45,
            "label": "Verlaagde afvoer",
        },
    ]

    # Define custom titles per NAME
    custom_titles = {
        "IJssel": "IJssel afvoer bij Olst per dag",
        "Neerslag": "Neerslag op dagbasis",
        "Verdamping": "Verdamping op dagbasis",
        "Houtrib spuisluizen (Houtribdijk) aanvoer": "Spuidebiet Houtribsluizen per dag (richting IJsselmeer)",
        "Houtrib spuisluizen (Houtribdijk) afvoer": "Spuidebiet Houtribsluizen per dag (richting Markermeer)",
        "Krabbersgat spuisluizen (Houtribdijk) aanvoer": "Spuidebiet Krabbersgatsluizen per dag (richting IJsselmeer)",
        "Krabbersgat spuisluizen (Houtribdijk) afvoer": "Spuidebiet Krabbersgatsluizen per dag (richting Markermeer)",
        "Stevin spuisluizen (Afsluitdijk)": "Spuidebiet Stevinsluizen per dag (richting Waddenzee)",
        "Lorentz spuisluizen (Afsluitdijk)": "Spuidebiet Lorentzsluizen per dag (richting Waddenzee)",
        "Berging IJsselmeer": "Berging IJsselmeer\nm.b.v. gew. gem. waterhoogte van 4 locaties in cm t.o.v. NAP",
    }

    files = result_df["NAME"].unique()
    if locations_to_include:
        files = [f for f in files if f in locations_to_include]

    for file in files:
        file_data = (
            result_df[result_df["NAME"] == file]
            .sort_values("DATUM")
            .reset_index(drop=True)
        )
        fig, ax1 = plt.subplots(figsize=(10, 5))

        if file == "Berging IJsselmeer":
            ax1.plot(
                file_data["DATUM"],
                file_data["WAARDE"],
                linewidth=1,
                color="lightblue",
                label="Berging",
                zorder=1,
            )

            ax1.set_title(custom_titles.get(file, file))
            ax1.set_ylim(None, None)  # Let matplotlib auto-scale for Berging
            ax1.set_ylabel("Berging in miljoen m³")
        else:
            # Apply flow category spans only for IJssel
            if file == "IJssel":
                for category in flow_categories:
                    ax1.axhline(
                        category["ymin"],
                        color=category["color"],
                        alpha=category["alpha"],
                        linestyle="--",
                        linewidth=1.5,
                    )
                    label_text = category.get("label", category.get("name", ""))
                    if label_text:
                        # Anchor y at the edge matching the vertical alignment so
                        # va="top"/"bottom" is meaningful (midpoint makes va a no-op,
                        # and ymax=inf would push the label off-screen).
                        if category["va"] == "top":
                            y_pos = category["ymax"] - 20
                        elif category["va"] == "bottom":
                            y_pos = category["ymin"] + 20
                        else:
                            y_pos = category["ymin"]
                        ax1.text(
                            0.01,
                            y_pos,
                            label_text,
                            transform=ax1.get_yaxis_transform(),
                            va=category["va"],
                            ha="left",
                            fontsize=9,
                            color=category["color"],
                            zorder=2,
                            bbox=dict(
                                boxstyle="round,pad=0.3",
                                facecolor="white",
                                edgecolor=category["color"],
                                alpha=0.8,
                            ),
                        )

            if file in outflows:
                file_data["WAARDE"] = -file_data["WAARDE"]
                ax1.plot(
                    file_data["DATUM"],
                    file_data["WAARDE"],
                    linewidth=0.9,
                    color="#005A9C",
                    label="Debiet",
                    zorder=1,
                )

            else:
                file_data["WAARDE"] = file_data["WAARDE"]
                ax1.plot(
                    file_data["DATUM"],
                    file_data["WAARDE"],
                    linewidth=0.8,
                    color="#005A9C",
                    label="Debiet",
                    zorder=1,
                )

            ax1.set_title(custom_titles.get(file, f"Debiet voor {file} IJsselmeer"))
            ax1.set_ylabel("Debiet in m³/s")
            ax1.axhline(y=0, color="grey", linestyle="-", linewidth=1, alpha=0.7)
        ax1.grid(True, alpha=0.25)
        if file == "IJssel":
            ax1.set_ylim(0, file_data["WAARDE"].max() * 1.1)  # Add 10% headroom
            ax1.set_xlim(pd.Timestamp("2017-02-01"), pd.Timestamp("2023-01-01"))
            # Format monthly datetime x-axis cleanly
            import matplotlib.dates as mdates
    
            _n_months = file_data["DATUM"].dt.to_period("M").nunique()
            _month_interval = next((d for d in (1, 2, 3, 4, 6, 12) if _n_months / d <= 12), 12)
            ax1.xaxis.set_major_locator(
                mdates.MonthLocator(bymonth=range(1, 13, _month_interval))
            )
            ax1.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(
                    ax1.xaxis.get_major_locator(),
                    formats=["%Y", "%b\n ", "%d", "%H:%M", "%H:%M", "%S.%f"],
                    zero_formats=["", "%b\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
                    show_offset=False,
                )
            )
            _xticks = ax1.get_xticks()
            if len(_xticks) > 1:
                ax1.set_xticks(_xticks[1:])
            plt.setp(ax1.get_xticklabels(), ha="center")
        else:
            ax1.set_xlim(pd.Timestamp("2017-12-31"), pd.Timestamp("2023-01-01"))
            # Format monthly datetime x-axis cleanly
            import matplotlib.dates as mdates
    
            _n_months = file_data["DATUM"].dt.to_period("M").nunique()
            _month_interval = next((d for d in (1, 2, 3, 4, 6, 12) if _n_months / d <= 12), 12)
            ax1.xaxis.set_major_locator(
                mdates.MonthLocator(bymonth=range(1, 13, _month_interval))
            )
            ax1.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(
                    ax1.xaxis.get_major_locator(),
                    formats=["%Y", "%b\n ", "%d", "%H:%M", "%H:%M", "%S.%f"],
                    zero_formats=["", "%b\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
                    show_offset=False,
                )
            )
            plt.setp(ax1.get_xticklabels(), ha="center")
        ax1.ticklabel_format(style="plain", axis="y")

        

        _valid_dates = file_data.loc[file_data["WAARDE"].notna(), "DATUM"]
        _min_date = _valid_dates.min()
        _max_date = _valid_dates.max()
        _min_str = _min_date.strftime("%d-%m-%Y") if pd.notna(_min_date) else "?"
        _max_str = _max_date.strftime("%d-%m-%Y") if pd.notna(_max_date) else "?"
        ax1.text(
            0.99,
            0.01,
            f"Data: {_min_str} t/m {_max_str} | Deltares | {TODAY}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="grey",
            alpha=0.7,
        )

        plt.tight_layout()
        file_clean = (
            file.replace(" ", "_")
            .replace("/", "-")
            .replace(",", "_")
            .replace("(", "_")
            .replace(")", "_")
            .replace(":", "_")
            .replace(";", "_")
            .replace(".", "_")
            .replace("-", "_")
        )
        file_clean = file_clean.replace("__", "_")
        fig_name_clean = f"carlijn_debiet_YMR_{file_clean}_{TODAY}.png"
        fig_name_clean = fig_name_clean.replace("__", "_")
        output_dir = Path(
            rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\debiet"
        )
        output_path = output_dir / fig_name_clean
        print(f"Saving figure to: {output_path}")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.show()

    # Plot sum of all other locations
    other_locations = [
        name for name in result_df["NAME"].unique() if name not in locations_to_include
    ]
    # print(f"Other locations to sum and plot: {other_locations}")
    if other_locations:
        # Create a copy and apply sign adjustment based on outflows
        other_filtered = result_df[result_df["NAME"].isin(other_locations)].copy()
        other_filtered["WAARDE_ADJUSTED"] = other_filtered.apply(
            lambda row: -row["WAARDE"] if row["NAME"] in outflows else row["WAARDE"],
            axis=1,
        )

        other_data = (
            other_filtered.groupby("DATUM")["WAARDE_ADJUSTED"]
            .sum()
            .reset_index(name="WAARDE")
            .sort_values("DATUM")
        )

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(
            other_data["DATUM"],
            other_data["WAARDE"],
            linewidth=0.8,
            color="#005A9C",
            label="Afvoer",
            zorder=1,
        )

        ax1.set_title("Overige regionale watervraag (totaal overige termen)")
        ax1.set_ylabel("Debiet in m³/s")
        ax1.axhline(y=0, color="grey", linestyle="-", linewidth=1, alpha=0.7)
        ax1.grid(True, alpha=0.25)
        ax1.set_xlim(pd.Timestamp("2017-12-31"), pd.Timestamp("2023-01-01"))
        ax1.ticklabel_format(style="plain", axis="y")
        # Format monthly datetime x-axis cleanly

        _n_months = file_data["DATUM"].dt.to_period("M").nunique()
        _month_interval = next((d for d in (1, 2, 3, 4, 6, 12) if _n_months / d <= 12), 12)
        ax1.xaxis.set_major_locator(
            mdates.MonthLocator(bymonth=range(1, 13, _month_interval))
        )
        ax1.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(
                ax1.xaxis.get_major_locator(),
                formats=["%Y", "%b\n ", "%d", "%H:%M", "%H:%M", "%S.%f"],
                zero_formats=["", "%b\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
                show_offset=False,
            )
        )
        plt.setp(ax1.get_xticklabels(), ha="center")

        _valid_dates = other_data.loc[other_data["WAARDE"].notna(), "DATUM"]
        _min_date = _valid_dates.min()
        _max_date = _valid_dates.max()
        _min_str = _min_date.strftime("%d-%m-%Y") if pd.notna(_min_date) else "?"
        _max_str = _max_date.strftime("%d-%m-%Y") if pd.notna(_max_date) else "?"
        ax1.text(
            0.99,
            0.01,
            f"Data: {_min_str} t/m {_max_str} | Deltares | {TODAY}",
            transform=ax1.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="grey",
            alpha=0.7,
        )

        plt.tight_layout()
        output_dir = Path(
            rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\debiet"
        )
        fig.savefig(
            output_dir / f"carlijn_debiet_overig_YMR_{TODAY}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
# %%

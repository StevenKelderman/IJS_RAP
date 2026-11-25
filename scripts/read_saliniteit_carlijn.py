# %%
from pathlib import Path

import pandas as pd
import matplotlib.dates as mdates
import matplotlib as mpl

# Directory to search for .YMR files
search_dir = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\data_vanCarlijn\CNonequi\CNonequi"
# search_dir = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\data_vanCarlijn\BalansData\BalansData\YMR\Zout\Cnet"

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

    # Print min and max years
    min_year = result_df["DATUM"].dt.year.min()
    max_year = result_df["DATUM"].dt.year.max()
    print(f"\nData year range: {min_year} - {max_year}")

    # Optional: Save to CSV
    result_df.to_parquet(
        f"P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\3_processed\\zout_YMR_carlijn_{min_year}_{max_year}.parquet",
        index=False,
    )
    print(f"\nSaved to zout_YMR_carlijn_{min_year}_{max_year}.parquet")
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
    import datetime as dt
    import warnings

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np

    TODAY = dt.date.today().strftime("%Y%m%d")

    files = result_df["NAME"].unique()
    for file in files:
        print(f"Plotting for {file}...")
        file_data = (
            result_df[result_df["NAME"] == file]
            .sort_values("DATUM")
            .reset_index(drop=True)
        )
        fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=False, sharey=False)

        # # Reindex to a continuous daily range so missing dates create gaps
        # full_range = pd.date_range(
        #     file_data["DATUM"].min(),
        #     file_data["DATUM"].max(),
        #     freq="1MS",
        # )
        # reindexed = (
        #     pd.DataFrame({"DATUM": full_range})
        #     .merge(file_data[["DATUM", "WAARDE"]], on="DATUM", how="left")
        #     .sort_values("DATUM")
        # )

        ax.plot(
            file_data["DATUM"],
            file_data["WAARDE"],
            linewidth=0.8,
            color="#132D6B",
            label="Daily",
        )

        ax.set_title(f"{file} - Chloride concentratie")
        ax.set_ylabel("Concentratie mg/l")

        # Set y-axis limit based on data values (ignore threshold lines)
        data_max = file_data["WAARDE"].max()
        y_top = data_max * 1.25 if pd.notna(data_max) and data_max > 0 else 1.0

        # Define flow categories for colored threshold lines
        flow_categories = [
            {
                "ymin": 16_500,
                "ymax": 100_000,
                "color": "#0F0F0D",
                "va": "bottom",
                "ls": "-.",
                "lw": 1.5,
                "label": "zout",
            },
            {
                "ymin": 10_000,
                "ymax": 16_500,
                "color": "#161947",
                "va": "bottom",
                "ls": "-.",
                "lw": 1.5,
                "label": "Sterk brak tot zout",
            },
            {
                "ymin": 3000,
                "ymax": 10_000,
                "color": "#1B4F9C",
                "va": "bottom",
                "ls": "-.",
                "lw": 1,
                "label": "Brak",
            },
            {
                "ymin": 1000,
                "ymax": 3000,
                "color": "#4FA3D1",
                "va": "bottom",
                "ls": "-.",
                "lw": 1,
                "label": "Zwak brak",
            },
            {
                "ymin": 300,
                "ymax": 1000,
                "color": "#7FD4E8",
                "va": "bottom",
                "ls": "--",
                "lw": 1.5,
                "label": "Licht brak",
            },
            {
                "ymin": 150,
                "ymax": 300,
                "color": "#A7DC14",
                "va": "bottom",
                "ls": "--",
                "lw": 1.5,
                "label": "Matig zoet",
            },
            {
                "ymin": 0,
                "ymax": 150,
                "color": "#1B7F1B",
                "va": "bottom",
                "ls": "--",
                "lw": 1,
                "label": "Zoet",
            },
        ]

        # Set y-axis lower limit based on data value. If the data min is above
        # 1000, only show relevant categories by starting the axis at the
        # highest category threshold that is still below the data min.
        data_min = file_data["WAARDE"].min()
        y_bottom = 0.0
        if pd.notna(data_min) and data_min > 1000:
            thresholds_below = [
                cat["ymin"] for cat in flow_categories if cat["ymin"] < data_min
            ]
            if thresholds_below:
                y_bottom = (
                    max(thresholds_below) * 0.98
                )  # Slightly below the highest threshold
        ax.set_ylim(y_bottom, y_top)

        for category in flow_categories:
            # Skip thresholds that fall outside the data-based y range
            if category["ymin"] > y_top or category["ymin"] < y_bottom:
                continue
            ax.axhline(
                category["ymin"],
                color=category["color"],
                alpha=0.45,
                linestyle=category["ls"],
                linewidth=category["lw"],
                zorder=-10,
            )
            label_text = category.get("label", category.get("name", ""))
            if label_text:
                # Always print the label just above the threshold line, with
                # an offset that scales with the visible y range so it stays
                # meaningful regardless of the axis limits (including when
                # y_bottom > 1000).
                offset = (y_top - y_bottom) * 0.02
                y_pos = category["ymin"] + offset
                # Skip labels that would fall outside the data-based y range
                if y_pos > y_top or y_pos < y_bottom:
                    continue
                ax.text(
                    0.01,
                    y_pos,
                    label_text,
                    transform=ax.get_yaxis_transform(),
                    va="bottom",
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

        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # ax.spines[['top', 'right']].set_visible(False)

        _valid_dates = file_data.loc[file_data["WAARDE"].notna(), "DATUM"]
        _min_date = _valid_dates.min()
        _max_date = _valid_dates.max()
        _min_str = _min_date.strftime("%d-%m-%Y") if pd.notna(_min_date) else "?"
        _max_str = _max_date.strftime("%d-%m-%Y") if pd.notna(_max_date) else "?"
        ax.text(
            0.99,
            0.99,
            f"Data: {_min_str} t/m {_max_str} | Deltares | {TODAY}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="grey",
            alpha=0.7,
        )

        ax.set_xlim(dt.datetime(2017, 6, 30), dt.datetime(2026, 4, 1))
        xlim_min = dt.datetime(2017, 6, 30)
        xlim_max = dt.datetime(2023, 1, 1)
        ax.set_xlim(xlim_min, xlim_max)

        # Check if no data within xlim but data after xlim_max
        data_in_xlim = file_data[
            (file_data["DATUM"] >= xlim_min) & (file_data["DATUM"] <= xlim_max)
        ]
        data_after_xlim = file_data[file_data["DATUM"] > xlim_max]
        if len(data_in_xlim) == 0 and len(data_after_xlim) > 0:
            ax.text(
                0.5,
                0.5,
                "Data available after 2023",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=18,
                color="grey",
                alpha=0.5,
            )

        ax.grid(True, alpha=0.25)
        # Format monthly datetime x-axis cleanly
        _n_months = file_data["DATUM"].dt.to_period("M").nunique()
        _month_interval = next((d for d in (1, 2, 3, 4, 6, 12) if _n_months / d <= 12), 12)
        ax.xaxis.set_major_locator(
            mdates.MonthLocator(bymonth=range(1, 13, _month_interval))
        )
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(
                ax.xaxis.get_major_locator(),
                formats=["%Y", "%b\n ", "%d", "%H:%M", "%H:%M", "%S.%f"],
                zero_formats=["", "%b\n%Y", "%b", "%b-%d", "%H:%M", "%H:%M"],
                show_offset=False,
            )
        )
        _xticks = ax.get_xticks()
        if len(_xticks) > 1:
            ax.set_xticks(_xticks[1:])
        plt.setp(ax.get_xticklabels(), ha="center")

        # Dummy legend with an invisible entry so the reserved space matches
        # plots that do have a (visible) legend
        dummy_handle = Line2D([], [], linestyle="none", marker="none")
        legend = ax.legend(
            [dummy_handle],
            [" "],
            fontsize=28,
            loc="upper left",
            title_fontsize=14,
            bbox_to_anchor=(1.0, 1.0),
        )
        legend.get_frame().set_alpha(0)
        for text in legend.get_texts():
            text.set_alpha(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        TODAY = 20260824

        file_clean = (
            file.replace(" ", "_")
            .replace("/", "-")
            .replace(",", "_")
            .replace("(", "_")
            .replace(")", "_")
        )
        file_clean = file_clean.replace("__", "_")
        fig_name_clean = f"carlijn_chloride_YMR_{file_clean}_{TODAY}.png"
        fig_name_clean = fig_name_clean.replace("__", "_")
        output_dir = Path(
            rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\saliniteit"
        )
        output_path = output_dir / fig_name_clean
        print(f"Saving figure to: {output_path}")
        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close()
# %%
# %%
# Plot WAARDE per DATUM for each FILE
# %%
# Plot monthly max and min WAARDE per location
# if dfs:
#     import matplotlib.pyplot as plt

#     files = result_df["NAME"].unique()
#     for file in files:
#         file_data = (
#             result_df[result_df["NAME"] == file]
#             .sort_values("DATUM")
#             .reset_index(drop=True)
#         )
#         file_data["Month"] = file_data["DATUM"].dt.month

#         monthly_stats = (
#             file_data.groupby("Month")["WAARDE"].agg(["min", "max"]).reset_index()
#         )

#         fig, ax = plt.subplots(figsize=(12, 6))

#         ax.plot(
#             monthly_stats["Month"],
#             monthly_stats["max"],
#             marker="o",
#             linestyle="-",
#             color="tab:red",
#             label="Max",
#         )
#         ax.plot(
#             monthly_stats["Month"],
#             monthly_stats["min"],
#             marker="o",
#             linestyle="-",
#             color="tab:blue",
#             label="Min",
#         )
#         ax.fill_between(
#             monthly_stats["Month"],
#             monthly_stats["min"],
#             monthly_stats["max"],
#             alpha=0.2,
#             color="tab:gray",
#         )

#         ax.set_xlabel("Month of Year")
#         ax.set_ylabel("Chloride gehalte in mg/l")
#         ax.legend()

#         plt.title(f"Monthly Min/Max Chloride gehalte in mg/l - {file}")
#         ax.grid()
#         plt.tight_layout()
#         plt.show()

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
#         file_data["4WeeksOfYear"] = file_data["DATUM"].dt.isocalendar().week // 4 + 1

#         weekly_stats = (
#             file_data.groupby("4WeeksOfYear")["WAARDE"]
#             .agg(["min", "max"])
#             .reset_index()
#         )

#         fig, ax = plt.subplots(figsize=(14, 6))

#         ax.plot(
#             weekly_stats["4WeeksOfYear"],
#             weekly_stats["max"],
#             marker="o",
#             linestyle="-",
#             color="tab:red",
#             label="Max",
#             markersize=3,
#         )
#         ax.plot(
#             weekly_stats["4WeeksOfYear"],
#             weekly_stats["min"],
#             marker="o",
#             linestyle="-",
#             color="tab:blue",
#             label="Min",
#             markersize=3,
#         )
#         ax.fill_between(
#             weekly_stats["4WeeksOfYear"],
#             weekly_stats["min"],
#             weekly_stats["max"],
#             alpha=0.2,
#             color="tab:gray",
#         )

#         ax.set_xlabel("Week of Year")
#         ax.set_ylabel("Chloride gehalte in mg/l")
#         ax.legend()

#         plt.title(f"Weekly Min/Max Chloride gehalte in mg/l - {file}")
#         ax.set_xlim(0, 15)
#         ax.grid(alpha=0.3)
#         plt.tight_layout()
#         plt.show()
# %%
file_data
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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\GIS\locaties_debiet.gpkg"
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

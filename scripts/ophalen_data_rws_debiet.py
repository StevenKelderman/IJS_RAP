# %%
# Imports
import datetime as dt
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ddlpy
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from contextily import add_basemap

# enabling debug logging so we can see what happens in the background
logging.basicConfig()
logging.getLogger("ddlpy").setLevel(logging.INFO)

# %%

PROJECT_DIR = Path(r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer")

DATA_OUTPUT_DIR = PROJECT_DIR / "analyse" / "data" / "4_output"

TODAY = dt.datetime.now().strftime("%Y%m%d")
# %%

locations = ddlpy.locations()

## also add code as a variable
locations["Code"] = locations.index
locations.index.name = "LocationCode"
locations_pl = pl.from_pandas(locations, include_index=True)
locations_pl
# %%
locations_pl = locations_pl.filter(
    (
        # (
        #     (pl.col("LocationCode").str.contains("ijsselmeer"))
        #     | (pl.col("LocationCode").str.contains("denoever"))
        #     | (pl.col("LocationCode").str.contains("kornwerderzand"))
        #     | (pl.col("LocationCode").str.contains("krabbersgat"))
        #     | (pl.col("LocationCode").str.contains("houtrib"))
        #     | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
        #     | (pl.col("LocationCode").str.contains("kampen.ijssel"))
        # )
        # &
        pl.col("Grootheid.Code") == "Q"
    )
)
# filter for water level stations
locations_pl
# %%
# Read geopackage and find points within

gpkg_path = PROJECT_DIR / "analyse" / "data" / "3_processed" / "IJSSELMEER5.gpkg"

# Read the geopackage
gdf_boundary = gpd.read_file(gpkg_path)
print(f"Read geopackage with {len(gdf_boundary)} features")
print(f"CRS: {gdf_boundary.crs}")
gdf_boundary.plot(figsize=(10, 10), edgecolor="black", facecolor="none", aspect="auto")
# %%

# Convert locations to GeoDataFrame
locations_gdf = gpd.GeoDataFrame(
    locations_pl.to_pandas(),
    geometry=gpd.points_from_xy(locations_pl["Lon"], locations_pl["Lat"]),
    crs="EPSG:4326",
)

# Reproject to match geopackage CRS if needed
if gdf_boundary.crs != locations_gdf.crs:
    locations_gdf = locations_gdf.to_crs(gdf_boundary.crs)

# Find points within boundary
points_within = gpd.sjoin(locations_gdf, gdf_boundary, how="inner", predicate="within")
print(f"Found {len(points_within)} stations within geopackage boundary")
points_within_pl = pl.from_pandas(
    points_within.drop(columns="geometry"), include_index=False
)

# %%
unique_location_codes = points_within_pl["LocationCode"].unique().to_list()
unique_location_codes.sort()
print(unique_location_codes)
# %%

bool_stations = locations.index.isin(
    [
        "denoever.waddenzee.voorhaven",
        "kornwerderzand.waddenzee.buitenhaven",
        # "dronten.roggebotsluis.spuikoker",
        # "dronten.roggebotsluis.vossemeer",
        # "genemuiden",
        # "krabbersgat.ijsselmeer",
        # "lelystad.houtribsluis.noord",
        # "lemmer.stroomkanaal.teroelsterkolk",
        # "lemmer.woudagemaal.stroomkanaal",
        # "stavoren.gemaalhoogland.buiten",
        # "stavoren.johanfrisokanaal",
        # "olst",
    ]
)
bool_grootheid = locations["Grootheid.Code"].isin(["Q"])
bool_proces = locations["ProcesType"].isin(["meting"])

selected = locations.loc[bool_stations & bool_grootheid & bool_proces]
selected
# %%
unique_names = selected["Naam"].unique()
unique_names
# %%


def fetch_measurements(args):
    idx, row, start_date, end_date = args
    station_name = row["Naam"]
    grootheid_code = row["Grootheid.Code"]
    parameter_wat_omschrijving = row["Parameter_Wat_Omschrijving"]
    print(
        f"({idx}) Getting measurements for {station_name} ({grootheid_code}): ({parameter_wat_omschrijving}) from {start_date} to {end_date}"
    )
    measurements = ddlpy.measurements(row, start_date, end_date)
    print(
        f"  Retrieved {len(measurements)} measurements for {station_name} ({grootheid_code})"
    )
    return measurements


# Date ranges: yearly periods from 1990 to 2026
date_ranges = [(f"{year}-01-01", f"{year}-12-31") for year in range(2018, 2027)]


# Helper function to create progress tracking filename
def get_location_progress_file(location_name):
    return (
        PROJECT_DIR
        / "analyse"
        / "data"
        / "1_raw"
        / "per_location_per_year"
        / "debiet"
        / f".progress_{location_name.replace(' ', '_').replace(',', '_')}.json"
    )


# Helper function to load completed date ranges for a location
def load_completed_ranges(location_name):
    progress_file = get_location_progress_file(location_name)
    if progress_file.exists():
        import json

        with open(progress_file, "r") as f:
            return json.load(f)
    return []


# Helper function to save completed date ranges for a location
def save_completed_ranges(location_name, completed_ranges):
    progress_file = get_location_progress_file(location_name)
    import json

    with open(progress_file, "w") as f:
        json.dump(completed_ranges, f)


print(
    f"Starting robust processing of {len(selected)} stations across {len(date_ranges)} date ranges..."
)

# Group by location first - this is the primary grouping
grouped_by_location = selected.groupby("Naam")

for location_name, location_group in grouped_by_location:
    print(f"\n{'='*80}")
    print(f"Processing Location: {location_name}")
    print(f"  Total parameters: {len(location_group)}")
    print(f"{'='*80}")

    # Load already completed date ranges for this location
    completed_ranges = load_completed_ranges(location_name)
    print(f"  Already completed date ranges: {completed_ranges}")

    for date_idx, (start_date, end_date) in enumerate(date_ranges):
        date_range_key = f"{start_date}_to_{end_date}"

        # Skip if already completed
        if date_range_key in completed_ranges:
            print(
                f"\n  [{date_idx + 1}/{len(date_ranges)}] Skipping date range {start_date} to {end_date} (already completed)"
            )
            continue

        print(
            f"\n  [{date_idx + 1}/{len(date_ranges)}] Processing date range {start_date} to {end_date}"
        )

        # Create output file for this location+daterange combination
        location_file = (
            PROJECT_DIR
            / "analyse"
            / "data"
            / "1_raw"
            / "per_location_per_year"
            / "debiet"
            / f"measurements_RWS_ijsselmeer_debiet_{location_name.replace(' ', '_').replace(',', '')}_{date_range_key[0:4]}_{TODAY}.parquet"
        )

        if location_file.exists():
            print(f"    File already exists, skipping: {location_file.name}")
            completed_ranges.append(date_range_key)
            save_completed_ranges(location_name, completed_ranges)
            continue

        # Process all parameters for this location + date range with threading
        n_params = len(location_group)
        measurements_list = []

        with ThreadPoolExecutor(max_workers=24) as executor:
            futures = {
                executor.submit(
                    fetch_measurements, (idx, row, start_date, end_date)
                ): idx
                for idx, (_, row) in enumerate(location_group.iterrows())
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result is not None and not result.empty:
                        measurements_list.append(result)
                    completed += 1
                    print(f"    Progress: {completed}/{n_params} parameters fetched")
                except Exception as e:
                    print(f"    Error fetching measurement: {e}")
                    completed += 1

        # Convert and save results for this location + date range
        if measurements_list:
            converted = [
                pl.from_pandas(df, include_index=True)
                for df in measurements_list
                if not df.empty
            ]
            if converted:
                combined = pl.concat(converted, how="diagonal_relaxed")
                combined.write_parquet(location_file)
                print(
                    f"    Saved {len(measurements_list)} parameters to {location_file.name}"
                )

        # Mark this date range as completed for this location
        completed_ranges.append(date_range_key)
        save_completed_ranges(location_name, completed_ranges)
        print(f"    Progress saved for {location_name}")

print(f"\n{'='*80}")
print(
    f"Completed processing of all {len(grouped_by_location)} locations across all date ranges"
)
print(f"{'='*80}")
# %%
# READ WADAR
measurements = (
    pl.scan_parquet(
        PROJECT_DIR
        / "analyse"
        / "data"
        / "1_raw"
        / "per_location_per_year"
        / "debiet"
        / "*.parquet",
        missing_columns="insert",
    )
    .filter(pl.col("WaarnemingMetadata.Kwaliteitswaardecode").cast(pl.Int32) <= 50)
    .collect()
)
measurements
# %%
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height

group_cols = [
    "Code",
    "Naam",
    "Grootheid.Code",
    "Parameter.Code",
    "Parameter_Wat_Omschrijving",
    "Eenheid.Code",
    "WaarnemingMetadata.Statuswaarde",
    "WaardeBewerkingsMethode.Omschrijving",
]

# Aggregate 10-minute data to daily means per station/parameter/height combination
measurements_daily = (
    measurements.with_columns(
        pl.col("time").cast(pl.Datetime, strict=False).dt.date().alias("time")
    )
    .group_by(group_cols + ["time"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("Meetwaarde.Waarde_Numeriek")
    )
    .sort(group_cols + ["time"])
)
measurements_daily
# %%
# Create overview for each Naam: time range and measurement count per year
overview = (
    measurements_daily.with_columns(
        pl.col("time").cast(pl.Date).dt.year().alias("year")
    )
    .group_by(["Naam", "year", "Parameter_Wat_Omschrijving"])
    .agg(pl.len().alias("count"))
    .pivot(
        index=["Naam", "Parameter_Wat_Omschrijving"],
        on="year",
        values="count",
        aggregate_function="sum",
    )
    .with_columns(pl.col("Naam").alias("Station"))
)

# Add time range per Naam
time_range = measurements_daily.group_by("Naam", "Parameter_Wat_Omschrijving").agg(
    pl.col("time").min().alias("start_time"),
    pl.col("time").max().alias("end_time"),
    pl.len().alias("total_measurements"),
)

overview_with_range = time_range.join(
    overview, on=["Naam", "Parameter_Wat_Omschrijving"], how="left"
).sort(["Naam", "Parameter_Wat_Omschrijving"])
# Sort year columns numerically
year_cols = sorted(
    [
        col
        for col in overview_with_range.columns
        if col
        not in [
            "Naam",
            "Parameter_Wat_Omschrijving",
            "start_time",
            "end_time",
            "total_measurements",
            "Station",
        ]
    ],
    key=int,
)
overview_with_range = overview_with_range.select(
    [
        "Naam",
        "Parameter_Wat_Omschrijving",
        "start_time",
        "end_time",
        "total_measurements",
    ]
    + year_cols
    + (["Station"] if "Station" in overview_with_range.columns else [])
)

overview_with_range.write_excel(
    DATA_OUTPUT_DIR / "overzicht_debiet_per_station_parameter.xlsx",
)
# %%
output_dir = (
    PROJECT_DIR
    / "analyse"
    / "data"
    / "2_external"
    / "measurements_RWS_ijsselmeer_debiet_per_combo"
)
output_dir.mkdir(parents=True, exist_ok=True)


def safe_part(value):
    value_str = "NA" if value is None else str(value)
    value_str = value_str.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_.-]", "", value_str)


group_keys = measurements_daily.select(group_cols).unique().sort(group_cols)
print(f"Exporting {len(group_keys)} grouped CSV files to: {output_dir}")

for row in group_keys.iter_rows(named=True):
    filter_expr = (
        (pl.col("Naam") == row["Naam"])
        & (pl.col("Grootheid.Code") == row["Grootheid.Code"])
        & (pl.col("Eenheid.Code") == row["Eenheid.Code"])
    )

    filtered = measurements_daily.filter(filter_expr)

    out_name = (
        f"{safe_part(row['Naam'])}"
        f"_{safe_part(row['Grootheid.Code'])}"
        f"_{safe_part(row['Eenheid.Code'])}_daily_ddl_wq.csv"
    )
    filtered.write_csv(output_dir / out_name, separator=";")

print("Done exporting grouped CSV files.")
# %%
# Generate median, 90th percentile, and 10th percentile per Naam
percentile_stats = (
    measurements.group_by(
        [
            "Naam",
            "Grootheid.Code",
            "Parameter_Wat_Omschrijving",
            "Eenheid.Code",
            "WaardeBewerkingsMethode.Omschrijving",
        ]
    )
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().round(2).alias("mean"),
        pl.col("Meetwaarde.Waarde_Numeriek").median().round(2).alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).round(2).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).round(2).alias("p10"),
    )
    .sort(
        [
            "Naam",
            "Grootheid.Code",
            "Parameter_Wat_Omschrijving",
            "Eenheid.Code",
            "WaardeBewerkingsMethode.Omschrijving",
        ]
    )
)
percentile_stats
# %%
# PLot all

group_keys = (
    measurements_daily.select(["Naam", "Parameter_Wat_Omschrijving", "Eenheid.Code"])
    .unique()
    .sort(["Naam", "Parameter_Wat_Omschrijving", "Eenheid.Code"])
)
num_plots = len(group_keys)
fig, axes = plt.subplots(num_plots, 1, figsize=(16, 4 * num_plots), sharex=False)
if num_plots == 1:
    axes = [axes]

for idx, row in enumerate(group_keys.iter_rows(named=True)):
    naam = row["Naam"]
    param = row["Parameter_Wat_Omschrijving"]
    eenheid = row["Eenheid.Code"]
    filtered = measurements_daily.filter(
        (pl.col("Naam") == naam)
        & (pl.col("Parameter_Wat_Omschrijving") == param)
        & (pl.col("Eenheid.Code") == eenheid)
    ).sort("time")

    # Calculate cumulative debiet
    cumulative = filtered.with_columns(
        pl.col("Meetwaarde.Waarde_Numeriek").cum_sum().alias("Cumulative")
    )

    # Plot discharge on primary axis
    ax1 = axes[idx]
    ax1.plot(
        filtered["time"],
        filtered["Meetwaarde.Waarde_Numeriek"],
        linewidth=0.5,
        label="Discharge",
    )
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    ax1.set_title(f"Discharge for {naam}\n{param}")
    ax1.set_ylabel(f"Discharge ({eenheid})")
    ax1.set_xlim(dt.datetime(1989, 12, 31), dt.datetime(2026, 4, 1))
    ax1.axvspan(
        dt.datetime(2018, 1, 1), dt.datetime(2026, 1, 1), alpha=0.2, color="grey"
    )
    ax1.grid(True, alpha=0.3)

    # Plot cumulative on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(
        cumulative["time"],
        cumulative["Cumulative"],
        linewidth=0.5,
        color="green",
        label="Cumulative",
    )
    ax2.set_ylabel("Cumulative (m³)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")
    ax2.set_ylim(ax1.get_ylim()[0], None)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR / "figuren" / "debiet" / "discharge_and_cumulative_per_station.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Plot showing 1 year with each year colored differently (using monthly means)

sns.set_style("whitegrid")
colors = plt.cm.plasma(np.linspace(0.3, 1, 35))

unique_names = measurements_daily["Naam"].unique()

num_plots = len(unique_names)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(unique_names):
    data = measurements.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()

    # Calculate monthly means
    data_pd["year_month"] = data_pd["time"].dt.to_period("M")
    monthly_means = (
        data_pd.groupby("year_month")
        .agg({"Meetwaarde.Waarde_Numeriek": "median", "time": "first"})
        .reset_index()
    )

    # Extract year and month of year
    monthly_means["year"] = monthly_means["time"].dt.year
    monthly_means["month"] = monthly_means["time"].dt.month

    # Plot each year with different color
    for year_idx, year in enumerate(sorted(monthly_means["year"].unique())):
        year_data = monthly_means[monthly_means["year"] == year]
        axes[idx].plot(
            year_data["month"],
            year_data["Meetwaarde.Waarde_Numeriek"],
            label=str(year),
            color=colors[year_idx % len(colors)],
            linewidth=1.5,
            marker="o",
        )
    axes[idx].axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    axes[idx].set_xlabel("Maand")
    axes[idx].set_ylabel("Maandelijkse Mediaan Meetwaarde")
    axes[idx].set_title(
        f"Maandelijkse Mediaan Meetwaarde per Maand (1 jaar cyclus) \n{naam}"
    )
    axes[idx].set_xticks(range(1, 13))
    axes[idx].legend(title="Jaar", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR / "figuren" / "debiet" / "monthly_median_per_year.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Plot showing 1 year with each year colored differently (using monthly means)

sns.set_style("whitegrid")
colors = plt.cm.Greys(np.linspace(0.3, 1, 10))

unique_names = measurements["Naam"].unique()

num_plots = len(unique_names)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(unique_names):
    data = measurements.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()

    # Calculate monthly means
    data_pd["year_month"] = data_pd["time"].dt.to_period("M")
    monthly_means = (
        data_pd.groupby("year_month")
        .agg({"Meetwaarde.Waarde_Numeriek": "median", "time": "first"})
        .reset_index()
    )
    monthly_means["time"] = monthly_means["year_month"].dt.to_timestamp()

    # Extract year and month of year
    monthly_means["year"] = monthly_means["time"].dt.year
    monthly_means["month"] = monthly_means["time"].dt.month

    # Plot each year with different color
    for year_idx, year in enumerate(sorted(monthly_means["year"].unique())):
        year_data = monthly_means[monthly_means["year"] == year]
        color = colors[year_idx % len(colors)]

        # Calculate 95% confidence interval
        data_by_month = [
            data_pd[
                (data_pd["time"].dt.year == year) & (data_pd["time"].dt.month == m)
            ]["Meetwaarde.Waarde_Numeriek"].values
            for m in range(1, 13)
        ]
        ci_lower = [
            np.percentile(values, 2.5) if len(values) > 0 else np.nan
            for values in data_by_month
        ]
        ci_upper = [
            np.percentile(values, 97.5) if len(values) > 0 else np.nan
            for values in data_by_month
        ]

        axes[idx].plot(
            year_data["month"],
            year_data["Meetwaarde.Waarde_Numeriek"],
            label=str(year),
            color=color,
            linewidth=1.5,
            marker="o",
        )
        axes[idx].fill_between(
            range(1, 13),
            ci_lower,
            ci_upper,
            color=color,
            alpha=0.2,
        )

    axes[idx].set_xlabel("Maand")
    axes[idx].set_ylabel("Maandelijkse Mediaan Meetwaarde")
    axes[idx].set_title(
        f"Maandelijkse Mediaan Meetwaarde per Maand (1 jaar cyclus) - {naam}"
    )
    axes[idx].set_xticks(range(1, 13))
    axes[idx].legend(title="Jaar", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR / "figuren" / "debiet" / "monthly_median_with_ci_per_year.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Plot static map

locations_pl_coords = locations_pl.select(
    [
        "LocationCode",
        "Lat",
        "Lon",
        "Naam",
        "Grootheid.Code",
    ]
).unique()

# Create figure with standard projection
fig, ax = plt.subplots(figsize=(25, 15))

# Plot locations
locations_df = locations_pl_coords.to_pandas()

# Merge median Meetwaarde values with locations
# Merge median Meetwaarde values with locations
median_values = percentile_stats[
    ["Naam", "Grootheid.Code", "Eenheid.Code", "mean"]
].to_pandas()
locations_df = locations_df.merge(
    median_values,
    left_on=["Naam", "Grootheid.Code"],
    right_on=["Naam", "Grootheid.Code"],
    how="inner",
)

locations_df = locations_df.dropna(subset=["mean"])  # Drop rows without median values

scatter = ax.scatter(
    locations_df["Lon"],
    locations_df["Lat"],
    c=locations_df["mean"],
    cmap="viridis",
    s=200,
    alpha=0.6,
    edgecolors="k",
    zorder=5,
)
cbar = fig.colorbar(scatter, ax=ax, label="Mean Meetwaarde")  # was plt.colorbar(...)

# Add labels for each location + mean value
for idx, (_, row) in enumerate(locations_df.iterrows()):
    ha = ["left", "right"][idx % 2]
    va = ["top", "bottom"][(idx // 2) % 2]
    offset_x = 0.03 if ha == "left" else -0.03
    offset_y = 0.03 if va == "top" else -0.03
    ax.plot(
        [row["Lon"], row["Lon"] + offset_x],
        [row["Lat"], row["Lat"] + offset_y],
        color="k",
        linewidth=0.5,
        zorder=4,
    )
    ax.text(
        row["Lon"] + offset_x,
        row["Lat"] + offset_y,
        f"{row['Naam']}\n{row['Grootheid.Code']}\nmean: {row['mean']:.2f} {row['Eenheid.Code']}",
        fontsize=7,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="grey"
        ),
        zorder=6,
    )

# Add OpenStreetMap basemap
ax.set_xlim(4.55, 5.9)
ax.set_ylim(52.2, 53.4)
add_basemap(
    ax,
    crs="EPSG:4258",
    source="OpenStreetMap.Mapnik",
    zoom=10,
    alpha=0.5,
    attribution_size=8,
)
ax.set_xlabel("Lengtegraad")
ax.set_ylabel("Breedtegraad")
plt.title("Debiet Meetlocaties (OpenStreetMap)")
plt.savefig(
    DATA_OUTPUT_DIR / "figuren" / "debiet" / "debiet_locations_map.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Plot interactive map with leaflet (folium)

# Calculate center of map
center_lat = locations_df["Lat"].mean()
center_lon = locations_df["Lon"].mean()

# Create folium map
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="OpenStreetMap")

# Normalize mean values for color mapping
min_mean = locations_df["mean"].min()
max_mean = locations_df["mean"].max()
locations_df["mean_normalized"] = (locations_df["mean"] - min_mean) / (
    max_mean - min_mean
)

# Add markers for each location
for idx, (_, row) in enumerate(locations_df.iterrows()):
    # Color based on mean value (viridis colormap)
    color_val = row["mean_normalized"]
    # Viridis colormap approximation
    if color_val < 0.25:
        color = "purple"
    elif color_val < 0.5:
        color = "blue"
    elif color_val < 0.75:
        color = "green"
    else:
        color = "yellow"

    popup_text = f"""
    <b>{row["Naam"]}</b><br>
    Parameter: {row["Grootheid.Code"]}<br>
    Mean: {row["mean"]:.2f} {row["Eenheid.Code"]}
    """

    folium.CircleMarker(
        location=[row["Lat"], row["Lon"]],
        radius=10,
        popup=folium.Popup(popup_text, max_width=300),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2,
    ).add_to(m)

# Add a color scale legend
legend_html = """
<div style="position: fixed;
     bottom: 50px; right: 50px; width: 200px; height: 150px;
     background-color: white; border:2px solid grey; z-index:9999;
     font-size:12px; padding: 10px">
     <p style="margin: 0;"><b>Debiet Levels</b></p>
     <p style="margin: 5px 0;"><i style="background:purple;width:20px;height:20px;display:inline-block;"></i> Low</p>
     <p style="margin: 5px 0;"><i style="background:blue;width:20px;height:20px;display:inline-block;"></i> Low-Mid</p>
     <p style="margin: 5px 0;"><i style="background:green;width:20px;height:20px;display:inline-block;"></i> Mid-High</p>
     <p style="margin: 5px 0;"><i style="background:yellow;width:20px;height:20px;display:inline-block;"></i> High</p>
</div>
"""
# Save interactive map
m.save(DATA_OUTPUT_DIR / "figuren" / "debiet" / "interactive_map_debiet_locations.html")
print("Interactive map saved to: interactive_map_debiet_locations.html")
# %%
import pandas as pd

measurements_daily

measurements_daily_filt = measurements_daily.filter(
    (pl.col("time") >= pd.Timestamp("1990-01-01"))
    & (pl.col("time") <= pd.Timestamp("2026-12-31"))
)
measurements_daily_filt
# %%

# Plot daily max and min WAARDE per location (by day of year)
# Convert from m³/s to miljoen m³/d: multiply by 86.4
files = measurements_daily_filt["Naam"].unique()
for file in files:
    file_data = (
        measurements_daily_filt.filter(measurements_daily_filt["Naam"] == file)
        .sort("time")
        .to_pandas()
        .reset_index(drop=True)
    )
    file_data["Meetwaarde.Waarde_Numeriek"] = (
        file_data["Meetwaarde.Waarde_Numeriek"] * 0.0864
    )
    file_data["DayOfYear"] = file_data["time"].dt.dayofyear
    file_data["DateNoYear"] = file_data["time"].dt.strftime("%m-%d")

    daily_stats = (
        file_data.groupby("DayOfYear")["Meetwaarde.Waarde_Numeriek"]
        .agg(["min", "max"])
        .reset_index()
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

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, zorder=-10)
    ax.set_xlabel("Maand-dag (MM-DD)")
    ax.set_ylabel("Debiet in miljoen m³ per dag")
    ax.legend()

    plt.title(
        f"Dagelijkse Min/Max Debiet in miljoen m³ per dag - periode (1990-2026)\n{file}"
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
    plt.savefig(
        DATA_OUTPUT_DIR
        / "figuren"
        / "debiet"
        / f"daily_min_max_debiet_{file.replace(' ', '_')}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
# %%

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
import pandas as pd
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
        #     | (pl.col("LocationCode").str.contains("andijk"))
        #     | (pl.col("LocationCode").str.contains("waddenzee"))
        #     # | (pl.col("LocationCode").str.contains("markermeer"))
        #     # | (pl.col("LocationCode").str.contains("doovebalg"))
        #     | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
        #     | (pl.col("LocationCode").str.contains("kampen.ijssel"))
        #     | (pl.col("LocationCode").str.contains("ijssel.km"))
        # )
        # &
        (pl.col("ProcesType") == "meting") & (pl.col("Hoedanigheid.Code") == "Cl")
    )
    | (
        # (
        #     (pl.col("LocationCode").str.contains("ijsselmeer"))
        #     | (pl.col("LocationCode").str.contains("denoever"))
        #     | (pl.col("LocationCode").str.contains("kornwerderzand"))
        #     | (pl.col("LocationCode").str.contains("andijk"))
        #     | (pl.col("LocationCode").str.contains("waddenzee"))
        #     # | (pl.col("LocationCode").str.contains("markermeer"))
        #     # | (pl.col("LocationCode").str.contains("doovebalg"))
        #     | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
        #     | (pl.col("LocationCode").str.contains("kampen.ijssel"))
        #     | (pl.col("LocationCode").str.contains("ijssel.km"))
        # )
        # &
        (pl.col("ProcesType") == "meting") & (pl.col("Grootheid.Code") == "SALNTT")
    )
)  # filter for salinity stations
locations_pl
# %%
# Read geopackage and find points within

gpkg_path = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\GIS\IJSSELMEER_MARKERMEER.gpkg"

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
        "andijk",
        "vrouwezand",
        "vrouwezand.meetpaal",
        # "broekerhaven.recreatiepark",
        "denoever.ijsselmeer.binnenhaven",
        "denoever.ijsselmeer.stevinsluizen.zuid",
        "denoever.ijsselmeer.vaargeul",
        "denoever.spuisluis.binnen",
        "denoever.waddenzee.voorhaven.kolk",
        # 'doovebalg.oost',
        # "friesekust.ijsselmeer",
        # "ketelhaven.ketelmeer",
        # "kornwerd",
        "kornwerderzand.ijsselmeer.binnenhaven",
        "kornwerderzand.ijsselmeer.middelgronden",
        "kornwerderzand.ijsselmeer.spui",
        "kornwerderzand.ijsselmeer.spuihavenzuid",
        "kornwerderzand.ijsselmeer.vaargeul",
        "kornwerderzand.waddenzee.voorhaven.kolk",
        "amsterdam.schellingwouderbrug",
        "deblocqvankuffeler.hogevaart",
        # "lelystad.houtribhoek",
        # "lemmer.teroelsterkolk.buiten",
        # "makkum.sluis.buiten",
        # "oosterdijk",
        # "oudezeug",
        # "ramsdiep",
        # "rotterdamsehoek",
        # "rotterdamsehoek.ijsselmeer",
        # "stavoren.gemaalhoogland.buiten",
        # "steilebank",
        # "wieringen.ijsselmeer.oost",
    ]
)

bool_procestype = locations["ProcesType"].isin(["meting"])
bool_grootheid_CONC = locations["Grootheid.Code"].isin(["CONCTTE"])  # salinity (SALNTT)
bool_grootheid_SAL = locations["Grootheid.Code"].isin(["SALNTT"])  # salinity (SALNTT)
bool_parameter_CL = locations["Parameter.Code"].isin(["Cl"])
bool_parameter_SAL = locations["Parameter.Code"].isin(["NVT"])

selected = locations.loc[
    bool_stations & bool_procestype & (bool_grootheid_CONC & bool_parameter_CL)
    # | (bool_grootheid_SAL & bool_parameter_SAL)
]
selected

selected_pl = pl.from_pandas(selected, include_index=True)
selected_pl
# %%
unique_location_codes = selected_pl["LocationCode"].unique().to_list()
unique_location_codes

# %%
unique_location_codes = (
    selected_pl.select(["LocationCode", "Naam", "Omschrijving"])
    .unique()
    .sort("LocationCode")
)
unique_location_codes
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
date_ranges = [(f"{year}-01-01", f"{year}-12-31") for year in range(1990, 2027)]


# Helper function to create progress tracking filename
def get_location_progress_file(location_name):
    return (
        PROJECT_DIR
        / "analyse"
        / "data"
        / "1_raw"
        / "per_location_per_year"
        / "saliniteit"
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
            / "saliniteit"
            / f"measurements_RWS_ijsselmeer_saliniteit_{location_name.replace(' ', '_').replace(',', '')}_{date_range_key[0:4]}_{TODAY}.parquet"
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
        / "saliniteit"
        / "*.parquet",
        missing_columns="insert",
    )
    .filter(pl.col("WaarnemingMetadata.Kwaliteitswaardecode").cast(pl.Int32) <= 50)
    .collect(engine="streaming")
)
measurements
# %%
# Inspect raw Bemonsteringshoogte values before conversion
print(
    measurements.select("WaarnemingMetadata.Bemonsteringshoogte")
    .unique()
    .sort("WaarnemingMetadata.Bemonsteringshoogte")
)

# Convert bodemhoogte from string to int
measurements = measurements.with_columns(
    pl.col("WaarnemingMetadata.Bemonsteringshoogte")
    .cast(pl.Int64)
    .map_elements(
        lambda x: (
            x / 10_000
            if x is not None and x < -10_000
            else (x / 100 if x is not None else None)
        ),
        return_dtype=pl.Float64,
    )
    .alias("WaarnemingMetadata.Bemonsteringshoogte")
)

# Show the distinct converted heights so different depths are visible
print(
    measurements.select("WaarnemingMetadata.Bemonsteringshoogte")
    .unique()
    .sort("WaarnemingMetadata.Bemonsteringshoogte")
)
measurements
# %%

# measurements = measurements.filter(
#     pl.col("WaarnemingMetadata.Kwaliteitswaardecode")
#     .is_in(["00", "25"])
#     .alias("WaarnemingMetadata.Kwaliteitswaardecode")
# )

measurements.shape
# %%
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height
hoogte_col = "WaarnemingMetadata.Bemonsteringshoogte"
if (
    hoogte_col not in measurements.columns
    and "WaarnemingsHoogte" in measurements.columns
):
    hoogte_col = "WaarnemingsHoogte"

group_cols = ["Naam", "Grootheid.Code", "Eenheid.Code", hoogte_col]

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
    .group_by(
        [
            "Naam",
            "year",
            "Grootheid.Code",
            "Eenheid.Code",
            "WaarnemingMetadata.Bemonsteringshoogte",
        ]
    )
    .agg(pl.len().alias("count"))
    .pivot(
        index=[
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            "WaarnemingMetadata.Bemonsteringshoogte",
        ],
        on="year",
        values="count",
        aggregate_function="sum",
    )
    .with_columns(pl.col("Naam").alias("Station"))
)

# Add time range per Naam
time_range = measurements_daily.group_by(
    "Naam", "Grootheid.Code", "Eenheid.Code", "WaarnemingMetadata.Bemonsteringshoogte"
).agg(
    pl.col("time").min().alias("start_time"),
    pl.col("time").max().alias("end_time"),
    pl.len().alias("total_measurements"),
)

overview_with_range = time_range.join(
    overview,
    on=[
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
    ],
    how="left",
)
# Sort year columns numerically
year_cols = sorted(
    [
        col
        for col in overview_with_range.columns
        if col
        not in [
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            "WaarnemingMetadata.Bemonsteringshoogte",
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
        "Grootheid.Code",
        "Eenheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
        "start_time",
        "end_time",
        "total_measurements",
    ]
    + year_cols
    + (["Station"] if "Station" in overview_with_range.columns else [])
).sort(
    ["Naam", "Grootheid.Code", "Eenheid.Code", "WaarnemingMetadata.Bemonsteringshoogte"]
)

# overview_with_range.write_excel(
#     DATA_OUTPUT_DIR / "overzicht_saliniteit_per_station_parameter.xlsx",
# )
overview_with_range["Naam"].to_list()
# %%
output_dir = Path(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\2_external\measurements_RWS_saliniteit_per_combo"
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

    if row[hoogte_col] is None:
        filtered = measurements_daily.filter(filter_expr & pl.col(hoogte_col).is_null())
    else:
        filtered = measurements_daily.filter(
            filter_expr & (pl.col(hoogte_col) == row[hoogte_col])
        )

    out_name = (
        f"{safe_part(row['Naam'])}"
        f"_{safe_part(row['Grootheid.Code'])}"
        f"_{safe_part(row['Eenheid.Code'])}"
        f"_{safe_part(row[hoogte_col])}_daily_ddl_wq.csv"
    )
    filtered.write_csv(output_dir / out_name, separator=";")

print("Done exporting grouped CSV files.")
# %%
measurements_daily.group_by(
    [
        "Naam",
        "Grootheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
    ]
).agg(pl.len()).sort("len", descending=True)

measurements_daily
# %%
# Completeness check: years as columns, count of Meetwaarde values per location
completeness = (
    measurements_daily.with_columns(
        pl.col("time").cast(pl.Date).cast(pl.String).str.slice(0, 4).alias("Year")
    )
    .group_by(["Naam", "Year", "Grootheid.Code"])
    .agg(pl.col("Meetwaarde.Waarde_Numeriek").count().alias("count"))
    .pivot(
        on="Year",
        index=["Naam", "Grootheid.Code"],
        values="count",
        aggregate_function="first",
    )
    .sort(["Naam", "Grootheid.Code"])
)

completeness
# %%
# Generate median, 90th percentile, and 10th percentile per Naam
percentile_stats = (
    measurements_daily.group_by(
        [
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            "WaarnemingMetadata.Bemonsteringshoogte",
        ]
    )
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("mean"),
        pl.col("Meetwaarde.Waarde_Numeriek").median().alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).alias("p10"),
    )
    .sort(
        [
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            "WaarnemingMetadata.Bemonsteringshoogte",
        ]
    )
)
percentile_stats
# %%
measurements_daily
# %%
# Plot all

import matplotlib as mpl

unique_combinations = (
    measurements_daily.select(["Naam", "Grootheid.Code", "Eenheid.Code"])
    .unique()
    .sort(["Naam", "Grootheid.Code", "Eenheid.Code"])
)
# naam_list = measurements_daily.select("Naam").unique().sort("Naam")["Naam"].to_list()
# List of names to include
naam_list = [
    "Amsterdam, Schellingwouderbrug",
    "Andijk",
    # 'Broekerhaven, recreatiepark',
    "Den Oever IJsselmeer Vaargeul",
    "Den Oever Stevinsluizen zuid",
    "Den Oever, IJsselmeer binnenhaven",
    "Den Oever, Waddenzee voorhaven kolk",
    "Den Oever, spuisluis, binnen",
    # 'Friesekust IJsselmeer',
    # 'Kornwerd',
    "Kornwerderzand IJsselmeer Middelgronden",
    "Kornwerderzand IJsselmeer spui",
    "Kornwerderzand IJsselmeer vaargeul",
    "Kornwerderzand Spuihaven zuid",
    "Kornwerderzand, IJsselmeer binnenhaven",
    "Kornwerderzand, Waddenzee voorhaven kolk",
    # 'Lelystad, Houtribhoek',
    # 'Lemmer Teroelsterkolk buiten',
    # 'Makkum sluis buiten',
    # "deblocqvankuffeler.hogevaart",
    # 'Oosterdijk',
    # 'Oude Zeug',
    # 'Ramsdiep',
    # 'Rotterdamse Hoek',
    # 'Rotterdamsehoek IJsselmeer',
    # 'Stavoren gemaal Hoogland buiten',
    # 'Steile Bank',
    "Vrouwezand",
    "Vrouwezand, meetpaal",
    # 'Wieringen IJsselmeer oost'
]

grootheid_list = (
    measurements_daily.select("Grootheid.Code")
    .unique()
    .sort("Grootheid.Code")["Grootheid.Code"]
    .to_list()
)

# Plot each location individually
for naam in naam_list:
    num_cols = len(grootheid_list)
    fig, axes = plt.subplots(1, num_cols, figsize=(10, 5), sharex=False, sharey=False)

    if num_cols == 1:
        axes = [axes]
    else:
        axes = list(np.atleast_1d(axes).ravel())

    for col_idx, grootheid in enumerate(grootheid_list):
        filtered = measurements_daily.filter(
            (pl.col("Naam") == naam)
            & (pl.col("Grootheid.Code") == grootheid)
            & (pl.col("time") > dt.date(2017, 12, 31))
        ).sort("time")

        ax = axes[col_idx]
        if len(filtered) > 0:
            # Get unique sampling heights and assign colors
            unique_heights = (
                filtered.select("WaarnemingMetadata.Bemonsteringshoogte")
                .unique()
                .sort(by="WaarnemingMetadata.Bemonsteringshoogte")[
                    "WaarnemingMetadata.Bemonsteringshoogte"
                ]
                .to_list()
            )
            colors = plt.colormaps.get_cmap("RdYlBu")(
                np.linspace(0, 1, len(unique_heights))
            )
            height_color_map = {
                height: colors[i] for i, height in enumerate(unique_heights)
            }

            # Plot each sampling height with different color
            for height in unique_heights:
                height_data = filtered.filter(
                    pl.col("WaarnemingMetadata.Bemonsteringshoogte") == height
                ).sort("time")
                if len(height_data) > 0:
                    line_color = (
                        "#132D6B"
                        if len(unique_heights) == 1
                        else height_color_map[height]
                    )

                    times = height_data["time"].to_list()
                    values = height_data["Meetwaarde.Waarde_Numeriek"].to_list()

                    # Reindex to a continuous daily range so missing dates create
                    # gaps (NaN) in the solid line
                    full_range = pl.date_range(
                        height_data["time"].min(),
                        height_data["time"].max(),
                        interval="1d",
                        eager=True,
                    )
                    reindexed = (
                        pl.DataFrame({"time": full_range})
                        .join(
                            height_data.select(["time", "Meetwaarde.Waarde_Numeriek"]),
                            on="time",
                            how="left",
                        )
                        .sort("time")
                    )
                    full_times = reindexed["time"].to_list()
                    full_values = reindexed["Meetwaarde.Waarde_Numeriek"].to_list()

                    # Dotted bridging line connecting across gaps
                    # ax.plot(
                    #     times,
                    #     values,
                    #     linewidth=0.6,
                    #     linestyle=":",
                    #     color=line_color,
                    #     alpha=0.6,
                    # )
                    # Solid line with gaps where data is missing

                    ax.plot(
                        full_times,
                        full_values,
                        linewidth=0.8,
                        label=f"Height: {height}",
                        color=line_color,
                    )

            eenheid = (
                filtered.select("Eenheid.Code").unique()["Eenheid.Code"].to_list()[0]
                if len(filtered) > 0
                else ""
            )
            ax.set_title(f"{naam} - Chloride concentratie")
            ax.set_ylabel(f"Concentratie {eenheid}")

            ax.yaxis.set_major_formatter(
                mpl.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )

            # Set y-axis limit based on data values (ignore threshold lines)
            data_max = filtered["Meetwaarde.Waarde_Numeriek"].max()
            y_top = data_max * 1.25 if data_max is not None and data_max > 0 else 1.0

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
            data_min = filtered["Meetwaarde.Waarde_Numeriek"].min()
            y_bottom = 0.0
            if data_min is not None and data_min > 1000:
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

            _full_data = (
                measurements_daily.filter(pl.col("Naam") == naam)
                .select(["time", "Meetwaarde.Waarde_Numeriek"])
                .to_pandas()
            )
            _valid_dates = _full_data.loc[
                _full_data["Meetwaarde.Waarde_Numeriek"].notna(), "time"
            ]
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

            # # Shade every other year
            # if len(filtered) > 0:
            #     min_year = filtered["time"].min().year
            #     max_year = filtered["time"].max().year
            #     for year in range(min_year, max_year + 1):
            #         if year % 2 == 0:
            #             ax.axvspan(
            #                 dt.datetime(year, 1, 1),
            #                 dt.datetime(year + 1, 1, 1),
            #                 color="grey",
            #                 alpha=0.1,
            #             )

            ax.set_xlim(dt.datetime(2017, 4, 30), dt.datetime(2026, 4, 1))
            xlim_min = dt.datetime(2017, 4, 30)
            xlim_max = dt.datetime(2023, 1, 1)
            ax.set_xlim(xlim_min, xlim_max)

            # Check if no data within xlim but data after xlim_max
            data_in_xlim = measurements_daily.filter(
                (pl.col("time") >= xlim_min) & (pl.col("time") <= xlim_max)
            )
            data_after_xlim = measurements_daily.filter(pl.col("time") > xlim_max)
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

            # ax.axvspan(
            #     dt.datetime(2018, 1, 1),
            #     dt.datetime(2026, 1, 1),
            #     alpha=0.2,
            #     color="grey",
            # )

            ax.legend(
                fontsize=12,
                loc="upper left",
                title="Diepte (m NAP)",
                title_fontsize=14,
                bbox_to_anchor=(1.0, 1.0),
            )
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    # Save individual plot
    TODAY = "20260824"
    file_clean = (
        naam.replace(" ", "_")
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
    fig_name_clean = f"WADAR_chloride_{file_clean}_{TODAY}.png"
    fig_name_clean = fig_name_clean.replace("__", "_")
    output_dir = Path(
        rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\run_{TODAY}\figuren\saliniteit"
    )
    output_path = output_dir / fig_name_clean
    print(f"Saving figure to: {output_path}")
    plt.savefig(
        output_path,
        dpi=300,
    )
    plt.show()
    plt.close()

# %%
measurements_daily

# %%

# Plot seasonal median values (summer vs winter) for each station
sns.set_style("whitegrid")

num_plots = len(measurements_daily["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements_daily["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting seasonal median time series for station: {naam}"
    )
    data = measurements_daily.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()
    data_pd[hoogte_col] = data_pd[hoogte_col].astype(str)

    # Add year and month columns
    data_pd["year"] = data_pd["time"].dt.year
    data_pd["month"] = data_pd["time"].dt.month

    # Define seasons: summer (4-9), winter (10,11,12,1,2,3)
    # Winter months 10,11,12 belong to the year-1, months 1,2,3 belong to the year
    def assign_season_and_year(row):
        month = row["month"]
        year = row["year"]
        if month in [4, 5, 6, 7, 8, 9]:
            return "Summer", year
        elif month in [10, 11, 12]:
            return "Winter", year + 1
        elif month in [1, 2, 3]:
            return "Winter", year
        else:
            return "Other", year

    data_pd[["season", "year"]] = data_pd.apply(
        assign_season_and_year, axis=1, result_type="expand"
    )

    # Filter for summer and winter only
    seasonal_data = data_pd[data_pd["season"] != "Other"]

    # Calculate median per year and season
    seasonal_median = (
        seasonal_data.groupby(["year", "season", hoogte_col])[
            "Meetwaarde.Waarde_Numeriek"
        ]
        .median()
        .reset_index()
    )

    if seasonal_median.empty:
        axes[idx].set_title(
            f"Seasonal Median by Bemonsteringshoogte - {naam} (no data)"
        )
        axes[idx].grid(True, alpha=0.3)
        continue

    sns.lineplot(
        data=seasonal_median,
        x="year",
        y="Meetwaarde.Waarde_Numeriek",
        hue=hoogte_col,
        style="season",
        markers=True,
        dashes=False,
        ax=axes[idx],
    )

    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Meetwaarde (Median)")
    axes[idx].set_title(f"Seasonal Median by Bemonsteringshoogte - {naam}")
    year_min = int(seasonal_median["year"].min())
    year_max = int(seasonal_median["year"].max())
    axes[idx].set_xlim(year_min - 0.5, year_max + 0.5)
    axes[idx].legend(title="Hoogte / Season")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "seasonal_median_meetwaarde_per_station.png",
    dpi=300,
)
plt.show()

# %%

# Plot median values for each station using monthly medians to smooth the data
sns.set_style("whitegrid")

num_plots = len(measurements_daily["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements_daily["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting monthly median time series for station: {naam}"
    )
    data = measurements_daily.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()
    data_pd[hoogte_col] = data_pd[hoogte_col].astype(str)

    # Calculate monthly median values
    data_pd["year_month"] = data_pd["time"].dt.to_period("M")
    monthly_median = (
        data_pd.groupby(["year_month", hoogte_col])["Meetwaarde.Waarde_Numeriek"]
        .median()
        .reset_index()
    )
    monthly_median["year_month"] = monthly_median["year_month"].dt.to_timestamp()

    if monthly_median.empty:
        axes[idx].set_title(f"Meetwaarde Median per Month - {naam} (no data)")
        axes[idx].grid(True, alpha=0.3)
        continue

    sns.lineplot(
        data=monthly_median,
        x="year_month",
        y="Meetwaarde.Waarde_Numeriek",
        hue=hoogte_col,
        marker="o",
        ax=axes[idx],
    )
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Meetwaarde (Median)")
    axes[idx].set_title(f"Meetwaarde Median per Month by Bemonsteringshoogte - {naam}")
    axes[idx].tick_params(axis="x", rotation=45)
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2026, 4, 1))
    axes[idx].legend(title="Bemonsteringshoogte")

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "monthly_median_meetwaarde_per_station.png",
    dpi=300,
)
plt.show()

# %%

measurements_a = measurements_daily.with_columns(
    pl.col("time").dt.strftime("%Y-%m").alias("year_month")
)
measurements_a = measurements_a.with_columns(
    pl.col("time").dt.strftime("%Y-%W").alias("year_week")
)

measurements_a = measurements_a.filter(pl.col("Grootheid.Code") == "CONCTTE")

measurements_a
# %%
# Create combinations of Naam and Grootheid.Code
combinations = (
    measurements_a.select(
        [
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            # "WaarnemingMetadata.Bemonsteringshoogte",
        ]
    )
    .unique()
    .sort(
        [
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            # "WaarnemingMetadata.Bemonsteringshoogte",
        ]
    )
    .to_dicts()
)
combinations
# %%
# Plot boxplot for each station by month
sns.set_style("darkgrid")

num_plots = len(combinations)

fig, axes = plt.subplots(
    num_plots, 1, figsize=(14, 4 * num_plots), sharex=True, sharey=False
)

if num_plots == 1:
    axes = [axes]

plot_idx = 0
for combo in combinations:
    naam = combo["Naam"]
    grootheid_code = combo["Grootheid.Code"]
    eenheid_code = combo["Eenheid.Code"]
    print(
        f"({plot_idx + 1}/{num_plots}) Plotting monthly boxplot for station: {naam}, Grootheid: {grootheid_code}"
    )
    data = measurements_a.filter(
        (pl.col("Naam") == naam)
        & (pl.col("Grootheid.Code") == grootheid_code)
        & (pl.col("Eenheid.Code") == eenheid_code)
        # & (pl.col("WaarnemingMetadata.Kwaliteitswaardecode").is_in(["00", "03"]))
    )

    if len(data) == 0:
        plot_idx += 1
        continue

    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()

    # Remove rows with NaN values in the columns we're plotting
    data_pd = data_pd.dropna(subset=["year_month", "Meetwaarde.Waarde_Numeriek"])

    data_pd_grouped = (
        data_pd.groupby(["year_month", "WaarnemingMetadata.Bemonsteringshoogte"])[
            "Meetwaarde.Waarde_Numeriek"
        ]
        .median()
        .reset_index()
    )
    data_pd_grouped["year_month"] = pd.to_datetime(data_pd_grouped["year_month"])
    data_pd_grouped = data_pd_grouped.sort_values(
        [
            "year_month",
            "WaarnemingMetadata.Bemonsteringshoogte",
        ],
        ascending=[True, True],
    )

    # Create boxplot
    # sns.boxplot(
    #     data=data_pd,
    #     x="year_week",
    #     y="Meetwaarde.Waarde_Numeriek",
    #     ax=axes[plot_idx],
    #     # notch=True,
    #     saturation=0.7,
    #     # showcaps=False,
    #     # flierprops={"marker": ""},
    #     # boxprops={"facecolor": (0.3, 0.5, 0.7, 0.5)},
    #     # medianprops={"color": "r", "linewidth": 2},
    # )
    sns.lineplot(
        data=data_pd_grouped,
        x="year_month",
        y="Meetwaarde.Waarde_Numeriek",
        ax=axes[plot_idx],
        markers=True,
        hue="WaarnemingMetadata.Bemonsteringshoogte",
        palette="coolwarm",
    )
    axes[plot_idx].set_xlabel("Month")
    axes[plot_idx].set_ylabel(f"Meetwaarde [{eenheid_code}]")
    # q3 = data_pd["Meetwaarde.Waarde_Numeriek"].quantile(0.75)
    # q2 = data_pd["Meetwaarde.Waarde_Numeriek"].quantile(0.5)
    # q_mean = data_pd["Meetwaarde.Waarde_Numeriek"].mean()
    # q1 = data_pd["Meetwaarde.Waarde_Numeriek"].quantile(0.25)
    # iqr = q3 - q1
    # y_max = (q3 + 1.5 * iqr) * 1.2
    # y_max = q_mean * 1.2
    # if np.isfinite(y_max) and y_max > 0:
    #     axes[plot_idx].set_ylim(0, y_max)
    axes[plot_idx].set_title(
        f"Meetwaarde Distribution per Month - {naam}\n({grootheid_code}  {eenheid_code})"  # {float(bemonsteringshoogte) / 100}m,
    )
    axes[plot_idx].tick_params(axis="x", rotation=45)

    plot_idx += 1

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "monthly_median_meetwaarde_boxplot_per_station.png",
    dpi=300,
)
plt.show()
# %%
# Plot showing 1 year with each year colored differently (using monthly means)
sns.set_style("whitegrid")

colors = plt.cm.plasma(np.linspace(0.3, 1, 10))

num_plots = len(measurements_daily["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements_daily["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting monthly median time series with yearly colors for station: {naam}"
    )
    data = measurements_daily.filter(pl.col("Naam") == naam)
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
        axes[idx].plot(
            year_data["month"],
            year_data["Meetwaarde.Waarde_Numeriek"],
            label=str(year),
            color=colors[year_idx % len(colors)],
            linewidth=1.5,
            marker="o",
        )

    axes[idx].set_xlabel("Month")
    axes[idx].set_ylabel("Monthly Median Meetwaarde")
    axes[idx].set_title(f"Monthly Median Meetwaarde by Month (1 year cycle) - {naam}")
    axes[idx].set_xticks(range(1, 13))
    axes[idx].set_xlim(1, 12)
    axes[idx].legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "monthly_median_meetwaarde_by_month_yearly_colors_per_station.png",
    dpi=300,
)
plt.show()

# %%
# Plot showing 1 year with each year colored differently (using monthly means)
sns.set_style("whitegrid")

colors = plt.cm.plasma(np.linspace(0.3, 1, 10))

num_plots = len(measurements_daily["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements_daily["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting monthly median time series with yearly colors and confidence intervals for station: {naam}"
    )
    data = measurements_daily.filter(pl.col("Naam") == naam)
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

    axes[idx].set_xlabel("Month")
    axes[idx].set_ylabel("Monthly Median Meetwaarde")
    axes[idx].set_title(f"Monthly Median Meetwaarde by Month (1 year cycle) - {naam}")
    axes[idx].set_xticks(range(1, 13))
    axes[idx].legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "monthly_median_meetwaarde_by_month_yearly_colors_ci_per_station.png",
    dpi=300,
)
plt.show()
# %%
# Plot showing 1 subplot per year with each location plotted
sns.set_style("whitegrid")

colors = plt.cm.tab20(np.linspace(0, 1, 20))

measurements_daily = measurements_daily.with_columns(
    pl.col("time").dt.year().alias("year")
)
measurements_daily = measurements_daily.with_columns(
    pl.col("time").dt.month().alias("month")
)

# Get unique years and locations
unique_years = sorted(measurements_daily["year"].unique().to_list())
unique_names = sorted(measurements_daily["Naam"].unique().to_list())

num_plots = len(unique_years)
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))

if num_plots == 1:
    axes = [axes]

for year_idx, year in enumerate(unique_years):
    year_data = measurements_daily.filter(pl.col("year") == year)

    # Plot each location with different color
    for loc_idx, naam in enumerate(unique_names):
        location_data = year_data.filter(pl.col("Naam") == naam)

        if len(location_data) > 0:
            # Calculate monthly mean for each location
            monthly_mean = (
                location_data.group_by("month")
                .agg(pl.col("Meetwaarde.Waarde_Numeriek").mean())
                .sort("month")
            )

            axes[year_idx].plot(
                monthly_mean["month"],
                monthly_mean["Meetwaarde.Waarde_Numeriek"],
                label=naam,
                color=colors[loc_idx % len(colors)],
                linewidth=1.5,
                marker="o",
                markersize=4,
            )

    axes[year_idx].set_xlabel("Month")
    axes[year_idx].set_ylabel("Monthly Mean Meetwaarde")
    axes[year_idx].set_title(f"Monthly Mean Meetwaarde by Location - Year {year}")
    axes[year_idx].set_xticks(range(1, 13))
    axes[year_idx].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[year_idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "monthly_mean_meetwaarde_by_location_yearly_subplots.png",
    dpi=300,
)
plt.show()
# %%
locations_pl_coords = locations_pl.select(
    [
        "LocationCode",
        "Lat",
        "Lon",
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
    ]
).unique()

# Plot locations
locations_df = locations_pl_coords.to_pandas()

# Merge median Meetwaarde values with locations
median_values = percentile_stats[
    [
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
        "median",
    ]
].to_pandas()
locations_df = locations_df.merge(
    median_values,
    left_on=["Naam", "Grootheid.Code", "Eenheid.Code"],
    right_on=["Naam", "Grootheid.Code", "Eenheid.Code"],
    how="left",
)

locations_df = locations_df.dropna(subset=["median"])  # Drop rows without median values
locations_df
# %%
locations_df = locations_df.loc[locations_df["Grootheid.Code"] == "CONCTTE"]
# locations_df = locations_df.loc[locations_df["Grootheid.Code"] == "SALNTT"]
locations_df
# %%
# Plot static map

# Create figure with standard projection
fig, ax = plt.subplots(figsize=(25, 15))

scatter = ax.scatter(
    locations_df["Lon"],
    locations_df["Lat"],
    c=locations_df["median"],
    cmap="viridis",
    s=200,
    alpha=0.6,
    edgecolors="k",
    zorder=5,
)
cbar = fig.colorbar(scatter, ax=ax, label="Median Meetwaarde")  # was plt.colorbar(...)

# Add labels for each location + median value
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
        f"{row['Naam']}\n{row['Grootheid.Code']}\nmed: {row['median']:.2f} {row['Eenheid.Code']}\n{row['WaarnemingMetadata.Bemonsteringshoogte']}m",
        fontsize=7,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="grey"
        ),
        zorder=6,
    )

# Add OpenStreetMap basemap
ax.set_xlim(4.95, 5.7)
ax.set_ylim(52.5, 53.15)
add_basemap(
    ax,
    crs="EPSG:4258",
    source="OpenStreetMap.Mapnik",
    zoom=10,
    alpha=0.5,
    attribution_size=8,
)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Salinity Measurement Locations (OpenStreetMap)")
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "measurement_locations_CONCTTE_map.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
locations_df_pivot = locations_df.pivot_table(
    index=["Naam", "Grootheid.Code", "Eenheid.Code"],
    columns="WaarnemingMetadata.Bemonsteringshoogte",
    values="median",
    aggfunc="first",
    sort=True,
)
locations_df_pivot
# %%
locations_df
# %%
# Boxplots per location with sampling height and parameter differentiation
sns.set_style("whitegrid")

num_plots = len(measurements_daily["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements_daily["Naam"].unique().sort()):
    print(f"({idx + 1}/{num_plots}) Plotting boxplot for station: {naam}")
    data = measurements_daily.filter(pl.col("Naam") == naam)
    data_pd = data.to_pandas()

    # Convert height to string for categorical display
    data_pd[hoogte_col] = data_pd[hoogte_col].astype(str)

    # Sort by height for better visualization
    height_order = sorted(data_pd[hoogte_col].unique(), key=float)

    # Create boxplot with height on x-axis and hue for different parameters
    sns.boxplot(
        data=data_pd,
        x=hoogte_col,
        y="Meetwaarde.Waarde_Numeriek",
        hue="Grootheid.Code",
        ax=axes[idx],
        palette="Set2",
        order=height_order,
    )

    axes[idx].set_xlabel("Sampling Height (m)")
    axes[idx].set_ylabel("Meetwaarde")
    axes[idx].set_title(f"Measurement Distribution by Height and Parameter - {naam}")
    axes[idx].tick_params(axis="x", rotation=45)
    axes[idx].legend(title="Parameter", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "boxplot_meetwaarde_by_height_parameter_per_location.png",
    dpi=300,
)
plt.show()

# %%
# Boxplots per parameter, with all Naam + Bemonsteringshoogte combinations colored differently
sns.set_style("whitegrid")

unique_locations = sorted(measurements_daily["Naam"].unique().to_list())
unique_parameters = sorted(measurements_daily["Grootheid.Code"].unique().to_list())

num_params = len(unique_parameters)
fig, axes = plt.subplots(num_params, 1, figsize=(10, 6 * num_params))

if num_params == 1:
    axes = [axes]

for param_idx, param in enumerate(unique_parameters):
    print(f"({param_idx + 1}/{num_params}) Plotting boxplot for parameter: {param}")

    param_data = measurements_daily.filter(
        pl.col("Grootheid.Code") == param
    ).to_pandas()

    if len(param_data) == 0:
        axes[param_idx].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[param_idx].set_title(f"{param} - No data")
        continue

    # Create a label combining Naam and Bemonsteringshoogte
    param_data["Location_Height"] = (
        param_data["Naam"] + " @ " + param_data[hoogte_col].astype(str) + "m"
    )

    # Sort by location and height for better visualization
    param_data_sorted = param_data.sort_values(["Naam", hoogte_col])
    location_height_order = sorted(param_data_sorted["Location_Height"].unique())

    # Create palette for all location+height combinations
    palette = sns.color_palette("husl", n_colors=len(location_height_order))

    # Create boxplot
    sns.boxplot(
        data=param_data_sorted,
        y="Location_Height",
        x="Meetwaarde.Waarde_Numeriek",
        ax=axes[param_idx],
        palette=palette,
        order=location_height_order,
    )

    axes[param_idx].set_ylabel("Location @ Height")
    axes[param_idx].set_xlabel("Meetwaarde")
    axes[param_idx].set_title(f"Parameter: {param}")
    axes[param_idx].tick_params(axis="y", rotation=0)
    axes[param_idx].grid(True, alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "boxplot_per_parameter_all_locations_heights.png",
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

# Normalize median values for color mapping
min_median = locations_df["median"].min()
max_median = locations_df["median"].max()
locations_df["median_normalized"] = (locations_df["median"] - min_median) / (
    max_median - min_median
)

# Add markers for each location
for idx, (_, row) in enumerate(locations_df.iterrows()):
    # Color based on median value (viridis colormap)
    color_val = row["median_normalized"]
    # Viridis colormap approximation
    if color_val < 0.001:
        color = "purple"
    elif color_val < 0.01:
        color = "blue"
    elif color_val < 0.1:
        color = "green"
    else:
        color = "yellow"

    popup_text = f"""
    <b>{row["Naam"]}</b><br>
    Parameter: {row["Grootheid.Code"]}<br>
    Median: {row["median"]:.2f} {row["Eenheid.Code"]}<br>
    Height: {row["WaarnemingMetadata.Bemonsteringshoogte"]}m
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
     <p style="margin: 0;"><b>Salinity Levels</b></p>
     <p style="margin: 5px 0;"><i style="background:purple;width:20px;height:20px;display:inline-block;"></i> Low</p>
     <p style="margin: 5px 0;"><i style="background:blue;width:20px;height:20px;display:inline-block;"></i> Low-Mid</p>
     <p style="margin: 5px 0;"><i style="background:green;width:20px;height:20px;display:inline-block;"></i> Mid-High</p>
     <p style="margin: 5px 0;"><i style="background:yellow;width:20px;height:20px;display:inline-block;"></i> High</p>
</div>
"""
# Save interactive map
m.save(
    DATA_OUTPUT_DIR
    / "figuren"
    / "saliniteit"
    / "interactive_map_salinity_SALNTT_locations.html"
)
print("Interactive map saved to: interactive_map_salinity_SALNTT_locations.html")
# %%

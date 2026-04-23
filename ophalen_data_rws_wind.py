# %%
# Imports
import datetime as dt
import logging
import re
from pathlib import Path

import ddlpy
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import polars as pl

# enabling debug logging so we can see what happens in the background
logging.basicConfig()
logging.getLogger("ddlpy").setLevel(logging.INFO)

# %%
# Locations

locations = ddlpy.locations()

## also add code as a variable
locations["Code"] = locations.index
locations.index.name = "LocationCode"
locations_pl = pl.from_pandas(locations, include_index=True)
locations_pl

# %%
locations_pl = locations_pl.filter(
    # (
    # (pl.col("LocationCode").str.contains("ijsselmeer"))
    # | (pl.col("LocationCode").str.contains("denoever"))
    # | (pl.col("LocationCode").str.contains("kornwerderzand"))
    # | (pl.col("LocationCode").str.contains("krabbersgat"))
    # |
    # | (pl.col("LocationCode").str.contains("houtrib"))
    # | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
    # | (pl.col("LocationCode").str.contains("Makkum"))
    # | (pl.col("LocationCode").str.contains("Workum"))
    # | (pl.col("LocationCode").str.contains("Hindeloopen"))
    # | (pl.col("LocationCode").str.contains("Molkwerum"))
    # | (pl.col("LocationCode").str.contains("Stavoren"))
    # | (pl.col("LocationCode").str.contains("Mirds"))
    # | (pl.col("LocationCode").str.contains("Mirdum"))
    # | (pl.col("LocationCode").str.contains("Lemmer"))
    # | (pl.col("LocationCode").str.contains("Staverskade"))
    # | (pl.col("LocationCode").str.contains("Vrouwezand"))
    # )
    # &
    (pl.col("Compartiment.Code") == "LT")
    # & (pl.col("Grootheid.Code").str.contains("WINDSHD"))
    & (pl.col("Grootheid.Code").str.starts_with("W"))
)
# filter for water level stations
locations_pl
# %%
# Read geopackage and find points within

gpkg_path = r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\3_processed\IJSSELMEER5.gpkg"

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
        "broekerhaven.recreatiepark",
        "dehegegerzen.oudemirdum",
        "dehollepoarte.makkum",
        "denoever.zuiderhaven.strand",
        "dezwaan",
        "doovebalg.midden",
        "doovebalg.oost",
        "drontermeerdijk.km0p4",
        "enkhuizen.ijsselmeer.oost.meetpaal1",
        "enkhuizen.ijsselmeer.oost.meetpaal2",
        "enkhuizen.ijsselmeer.oost.meetpaal25",
        "enkhuizen.strand",
        "hindeloopen.badpaviljoenhindeloopen",
        "houtribdijk.ijsselmeer",
        "ijsselmeer.midden.meetpaal26",
        "ijsselmeer.wagenpad.zuid",
        "itsoal.workum",
        "kamperhoek.badstrand",
        "lelystad.houtribhoek",
        "lelystad.houtribhoek.strand",
        "lemsterstrand.lemmer",
        "markerwaarddijk.ijsselmeer.zuidwest.meetpaal37",
        "medemblik.vooroever.hetnesbos",
        "medemblik.vooroever.vlietsingelpoel",
        "mirnserklif.mirns",
        "molkwerum.molkwerum",
        "oudemirdum.ijsselmeer.noordoost.meetpaal9",
        "oudemirdum.ijsselmeer.noordoost.meetpaalfl05",
        "oudezeug.ijsselmeer.midden.noord.meetpaal47",
        "oudezeug.ijsselmeer.midden.noord.meetpaal48",
        "oudezeug.ijsselmeer.midden.noord.meetpaal49",
        "ramsdiep",
        "rotterdamsehoek.ijsselmeer",
        "rotterdamsehoek.meetpaalfl2",
        "schokkerhaven.strand",
        "stavoren.ijsselmeer",
        "stavoren.itsuderstrand",
        "steilebank",
        "strandnieuwkamperhoek",
        "strandstaversekade",
        "urk.ijsselmeer.oost.meetpaal2",
        "urk.ijsselmeer.zuid.meetpaal46",
        "vooroeverandijkkoopmanspolder",
        "vrouwezand",
        "wevershoof.onderdijkzwemstrand",
        "zeughoek",
        "zwolsehoek.badstrand",
    ]
)
bool_grootheid = locations["Grootheid.Code"].str.startswith("W")
bool_compartiment = locations["Compartiment.Code"] == "LT"

selected = locations.loc[bool_stations & bool_grootheid & bool_compartiment]
selected
# %%
unique_names = selected["Naam"].unique()
unique_names
# %%

from concurrent.futures import ThreadPoolExecutor, as_completed


def fetch_measurements(args):
    idx, index, row = args
    station_name = row["Naam"]
    grootheid = row["Grootheid.Code"]
    wat = row["Parameter_Wat_Omschrijving"]
    print(
        f"({idx + 1}/{len(selected)}) Getting measurements for {station_name} ({grootheid}): ({wat})"
    )
    print("  Start date: 2018-01-01, End date: 2026-03-31")
    measurements = ddlpy.measurements(row, "2018-01-01", "2026-03-31")
    print(
        f"  Retrieved {len(measurements)} measurements for {station_name} ({grootheid})"
    )
    return measurements


all_measurements = []
print(f"Starting processing of {len(selected)} stations...")
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [
        executor.submit(fetch_measurements, (idx, index, row))
        for idx, (index, row) in enumerate(selected.iterrows())
    ]
    for future in as_completed(futures):
        all_measurements.append(future.result())
print(f"Completed processing of all {len(selected)} stations")

# %%
converted_measurements = [
    pl.from_pandas(df, include_index=True) for df in all_measurements if not df.empty
]
if converted_measurements:
    measurements = pl.concat(converted_measurements)
else:
    measurements = pl.DataFrame()

measurements
# %%
# EXPORT TO PARQUET

TODAY = dt.datetime.now().strftime("%Y%m%d")

measurements.write_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_wind_snelheid_2018_2026_{TODAY}.parquet"
)
# %%
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height

group_cols = [
    "Naam",
    "Grootheid.Code",
    "Eenheid.Code",
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

output_dir = Path(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\2_external\measurements_RWS_wind_per_combo"
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
# PLot all

unique_combinations = (
    measurements_daily.select(["Naam", "Grootheid.Code", "Eenheid.Code"])
    .unique()
    .sort(["Naam", "Grootheid.Code", "Eenheid.Code"])
)
naam_list = measurements_daily.select("Naam").unique().sort("Naam")["Naam"].to_list()
grootheid_list = (
    measurements_daily.select("Grootheid.Code")
    .unique()
    .sort("Grootheid.Code")["Grootheid.Code"]
    .to_list()
)
num_rows = len(naam_list)
num_cols = len(grootheid_list)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows), sharex=True)
if num_rows == 1 and num_cols == 1:
    axes = [[axes]]
elif num_rows == 1:
    axes = [axes]
elif num_cols == 1:
    axes = [[ax] for ax in axes]

for naam_idx, naam in enumerate(naam_list):
    for col_idx, grootheid in enumerate(grootheid_list):
        filtered = measurements_daily.filter(
            (pl.col("Naam") == naam) & (pl.col("Grootheid.Code") == grootheid)
        ).sort("time")

        ax = axes[naam_idx][col_idx]
        if len(filtered) > 0:
            ax.plot(
                filtered["time"],
                filtered["Meetwaarde.Waarde_Numeriek"],
                linewidth=0.5,
                label="Discharge",
            )
            eenheid = (
                filtered.select("Eenheid.Code").unique()["Eenheid.Code"].to_list()[0]
                if len(filtered) > 0
                else ""
            )
            ax.set_title(f"Windspeed for {naam}\n({grootheid})")
            ax.set_ylabel(f"{grootheid} [{eenheid}]")
            ax.set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2026, 4, 1))
        ax.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\wind_and_cumulative_per_station.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
measurements
# %%
# Wind rose plot with locations on map
import numpy as np

# Get wind direction data
wind_data_all = measurements_daily.filter(
    (
        (pl.col("Grootheid.Code") == "WINDRTG")  # Wind direction code
        | (pl.col("Grootheid.Code") == "WINDSHD")  # Windspeed
        # & (pl.col("time").dt.year() == 2021)  # Year filter
    )
)

# Get location coordinates from original locations data
locations_with_coords = locations_pl.select(["Naam", "Lat", "Lon"]).unique()

# Merge wind data with coordinates
wind_data_with_coords = wind_data_all.join(locations_with_coords, on="Naam", how="left")

# Create interactive map
m = folium.Map(
    location=[52.5, 5.5],  # Center of IJsselmeer region
    zoom_start=9,
    tiles="OpenStreetMap",
)

unique_names = sorted(wind_data_all.select("Naam").unique().to_series().to_list())

# Create wind roses as base64 images and add them to map for each location
import base64
import io

# Calculate min and max wind speeds for normalization
all_wind_speeds = []
for naam in unique_names:
    wind_data = wind_data_with_coords.filter(pl.col("Naam") == naam)
    if len(wind_data) > 0:
        wind_speed_data = wind_data.filter(pl.col("Grootheid.Code") == "WINDSHD")
        mean_speed = (
            wind_speed_data["Meetwaarde.Waarde_Numeriek"].mean()
            if len(wind_speed_data) > 0
            else 0
        )
        all_wind_speeds.append(mean_speed)

min_speed = min(all_wind_speeds) if all_wind_speeds else 0
max_speed = max(all_wind_speeds) if all_wind_speeds else 1

# Color map for wind speeds
color_map = plt.colormaps.get_cmap("RdYlGn")

# Define folium color palette
folium_colors = ["green", "lightgreen", "orange", "red", "darkred"]

for naam in unique_names:
    wind_data = wind_data_with_coords.filter(pl.col("Naam") == naam)

    if len(wind_data) > 0:
        # Get coordinates
        lat = wind_data.select("Lat").unique()[0, 0]
        lon = wind_data.select("Lon").unique()[0, 0]

        # Get wind direction and speed data
        wind_direction_data = wind_data.filter(pl.col("Grootheid.Code") == "WINDRTG")
        wind_speed_data = wind_data.filter(pl.col("Grootheid.Code") == "WINDSHD")

        theta = np.deg2rad(wind_direction_data["Meetwaarde.Waarde_Numeriek"].to_numpy())

        # Create wind rose histogram
        bins = np.arange(0, 2 * np.pi, np.pi / 16)  # 32 bins (11.25 degrees each)
        counts, _ = np.histogram(theta, bins=bins)

        # Create popup with statistics
        valid_theta = theta[~np.isnan(theta)]
        if len(valid_theta) > 0:
            mean_dir = (
                np.degrees(
                    np.arctan2(
                        np.mean(np.sin(valid_theta)), np.mean(np.cos(valid_theta))
                    )
                )
                % 360
            )
        else:
            mean_dir = np.nan
        mean_speed = (
            wind_speed_data["Meetwaarde.Waarde_Numeriek"].mean()
            if len(wind_speed_data) > 0
            else 0
        )
        mean_dir_str = f"{mean_dir:.1f}°" if not np.isnan(mean_dir) else "N/A"
        popup_text = (
            f"{naam}<br>Mean Direction: {mean_dir_str}<br>N Obs: {len(wind_data)}"
        )

        # Normalize mean speed and determine color
        if max_speed > min_speed:
            normalized_speed = (mean_speed - min_speed) / (max_speed - min_speed)
        else:
            normalized_speed = 0.5
        color_idx = min(
            int(normalized_speed * len(folium_colors)), len(folium_colors) - 1
        )
        marker_color = folium_colors[color_idx]

        # Create small wind rose figure
        fig_rose = plt.figure(figsize=(4, 4))
        ax_rose = fig_rose.add_subplot(111, projection="polar")

        # Plot wind rose
        width = 2 * np.pi / len(bins)
        bars = ax_rose.bar(bins[:-1], counts, width=width, alpha=0.7, edgecolor="black")

        # Colorize bars
        colors = plt.colormaps.get_cmap("viridis")(counts / counts.max())
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)

        # Set labels
        ax_rose.set_theta_zero_location("N")
        ax_rose.set_theta_direction(-1)
        ax_rose.set_title(f"{naam}", pad=10)

        # Convert figure to base64 string
        buffer = io.BytesIO()
        fig_rose.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig_rose)

        # Create HTML for popup with image
        html = f"""
        <div style="width: 300px; text-align: center;">
            <b>{naam}</b><br>
            Mean Windspeed: {mean_speed:.2f} m/s <br>
            Mean Direction: {mean_dir:.1f}°<br>
            N Obs: {len(wind_data)}<br>
            <img src="data:image/png;base64,{image_base64}" style="width: 250px;">
        </div>
        """

        # Add marker with popup, colored by normalized wind speed
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(html, max_width=350),
            tooltip=naam,
            icon=folium.Icon(color=marker_color, icon="wind", prefix="fa"),
        ).add_to(m)

# Save map
m.save(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\wind_rose_on_maps.html"
)
print("Wind rose map saved")
# %%
# Wind rose plot for degree data (0-360) - rows: Naam locations, columns: seasons
import numpy as np

wind_data_all = measurements_daily.filter(
    pl.col("Grootheid.Code") == "WINDRTG"  # Wind direction code
).with_columns(
    month=pl.col("time").dt.month(),
    season=pl.when(pl.col("time").dt.month().is_in([12, 1, 2]))
    .then(pl.lit("Winter"))
    .when(pl.col("time").dt.month().is_in([3, 4, 5]))
    .then(pl.lit("Spring"))
    .when(pl.col("time").dt.month().is_in([6, 7, 8]))
    .then(pl.lit("Summer"))
    .otherwise(pl.lit("Fall")),
)

unique_names = sorted(wind_data_all.select("Naam").unique().to_series().to_list())
seasons = ["Winter", "Spring", "Summer", "Fall"]

# Create figure with rows=locations, columns=seasons
n_locations = len(unique_names)
n_seasons = len(seasons)
fig = plt.figure(figsize=(16, 4 * n_locations))

for season_idx, season in enumerate(seasons):
    wind_data = wind_data_all.filter(
        (pl.col("Naam") == naam) & (pl.col("season") == season)
    ).to_pandas()

    ax = plt.subplot(
        n_locations,
        n_seasons,
        naam_idx * n_seasons + season_idx + 1,
        projection="polar",
    )

    if len(wind_data) > 0:
        # Convert degrees to radians
        theta = np.deg2rad(wind_data["Meetwaarde.Waarde_Numeriek"].values)

        # Create wind rose histogram
        bins = np.arange(0, 2 * np.pi, np.pi / 16)  # 32 bins (11.25 degrees each)
        counts, _ = np.histogram(theta, bins=bins)

        # Plot bars
        width = 2 * np.pi / len(bins)
        bars = ax.bar(bins[:-1], counts, width=width, alpha=0.7, edgecolor="black")

        # Colorize bars
        colors = plt.colormaps.get_cmap("viridis")(counts / counts.max())
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)

    # Set labels
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title(f"{season}\n{naam}", pad=10)

plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\wind_rose_by_season.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Generate median, 90th percentile, and 10th percentile per Naam
percentile_stats = (
    measurements_daily.group_by(
        [
            "Naam",
            "Grootheid.Code",
            "Eenheid.Code",
            "WaardeBewerkingsMethode.Omschrijving",
        ]
    )
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("mean"),
        pl.col("Meetwaarde.Waarde_Numeriek").median().alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).alias("p10"),
    )
    .sort(
        "Naam", "Grootheid.Code", "Eenheid.Code", "WaardeBewerkingsMethode.Omschrijving"
    )
)
percentile_stats
# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

unique_combinations = measurements_daily.select(
    [
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        "WaardeBewerkingsMethode.Omschrijving",
    ]
).unique()

num_plots = len(unique_combinations)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots), sharex=True)

if num_plots == 1:
    axes = [axes]

for idx, combination in enumerate(unique_combinations.iter_rows(named=True)):
    naam = combination["Naam"]
    grootheid = combination["Grootheid.Code"]
    eenheid = combination["Eenheid.Code"]
    bewerking = combination["WaardeBewerkingsMethode.Omschrijving"]

    data = measurements_daily.filter(
        (pl.col("Naam") == naam)
        & (pl.col("Grootheid.Code") == grootheid)
        & (pl.col("Eenheid.Code") == eenheid)
        & (pl.col("WaardeBewerkingsMethode.Omschrijving") == bewerking)
    )
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()
    data_pd["Meetwaarde.Waarde_Numeriek_cumsum"] = data_pd[
        "Meetwaarde.Waarde_Numeriek"
    ].cumsum()

    ax2 = axes[idx].twinx()

    sns.lineplot(
        data=data_pd,
        x="time",
        y="Meetwaarde.Waarde_Numeriek_cumsum",
        ax=axes[idx],
    )

    sns.lineplot(
        data=data_pd,
        x="time",
        y="Meetwaarde.Waarde_Numeriek",
        ax=ax2,
        color="orange",
    )

    axes[idx].set_xlabel("Date")
    axes[idx].set_ylabel("Cumsum Meetwaarde")
    ax2.set_ylabel("Meetwaarde")
    axes[idx].set_title(f"Meetwaarde over Time - {naam} | {grootheid} | {bewerking}")
    axes[idx].tick_params(axis="x", rotation=45)
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2026, 4, 1))

plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\wind_and_cumulative_per_station_detailed.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Plot showing 1 year with each year colored differently (using monthly means)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
colors = plt.cm.plasma(np.linspace(0.3, 1, 10))

unique_names = measurements_daily["Naam"].unique()

num_plots = len(unique_names)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(unique_names):
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
    axes[idx].legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\monthly_median_per_year.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Plot showing 1 year with each year colored differently (using monthly means)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
colors = plt.cm.Greys(np.linspace(0.3, 1, 10))

unique_names = measurements_daily["Naam"].unique()

num_plots = len(unique_names)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(unique_names):
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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\monthly_median_with_ci_per_year.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%

measurements_daily
# %%
# X-Y plot for Houtribsluis locations
import matplotlib.pyplot as plt
import numpy as np

houtribsluis_data = measurements_daily.filter(
    pl.col("Naam").str.contains("Houtribsluis")
)

if len(houtribsluis_data) > 0:
    houtribsluis_df = houtribsluis_data.sort("time").to_pandas()

    fig, ax = plt.subplots(figsize=(14, 6))

    unique_names = houtribsluis_df["Naam"].unique()
    if len(unique_names) >= 2:
        data1 = houtribsluis_df[houtribsluis_df["Naam"] == unique_names[0]]
        data2 = houtribsluis_df[houtribsluis_df["Naam"] == unique_names[1]]
        data1["year"] = data1["time"].dt.year
        colors = data1["year"].apply(lambda x: plt.cm.Set2(x % 8))
        ax.scatter(
            data1["Meetwaarde.Waarde_Numeriek"],
            data2["Meetwaarde.Waarde_Numeriek"],
            marker="o",
            s=30,
            c=data1["year"],
            cmap="Set2",
            alpha=0.7,
        )
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label("Year")
        # Add linear trendline
        mask = (
            data1["Meetwaarde.Waarde_Numeriek"].notna()
            & data2["Meetwaarde.Waarde_Numeriek"].notna()
        )
        if mask.sum() > 1:
            z = np.polyfit(
                data1.loc[mask, "Meetwaarde.Waarde_Numeriek"],
                data2.loc[mask, "Meetwaarde.Waarde_Numeriek"],
                1,
            )
            p = np.poly1d(z)
            x_trend = np.linspace(
                data1["Meetwaarde.Waarde_Numeriek"].min(),
                data1["Meetwaarde.Waarde_Numeriek"].max(),
                100,
            )
            ax.plot(x_trend, p(x_trend), "r-", linewidth=2, label="Trendline")
        ax.set_xlabel(unique_names[0])
        ax.set_ylabel(unique_names[1])
        ax.set_title(
            f"Water Height Comparison - {unique_names[0]} vs {unique_names[1]}"
        )
    else:
        for naam in houtribsluis_df["Naam"].unique():
            data = houtribsluis_df[houtribsluis_df["Naam"] == naam]
            ax.plot(
                data["time"],
                data["Meetwaarde.Waarde_Numeriek"],
                label=naam,
                marker="o",
                markersize=3,
                linewidth=1,
            )
        ax.set_xlabel("Time")
        ax.set_ylabel("Meetwaarde")
        ax.set_title("Water Height (Meetwaarde) - Houtribsluis Locations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No Houtribsluis data found")

# %%
locations_pl_coords = locations_pl.select(
    [
        "LocationCode",
        "Lat",
        "Lon",
        "Naam",
        "Grootheid.Code",
    ]
).unique()
locations_pl_coords
# %%
percentile_stats
# %%
import matplotlib.pyplot as plt
from contextily import add_basemap

# Create figure with standard projection
fig, ax = plt.subplots(figsize=(25, 15))

# Plot locations
locations_df = locations_pl_coords.to_pandas()

# Merge median Meetwaarde values with locations
median_values = percentile_stats[
    ["Naam", "Grootheid.Code", "Eenheid.Code", "median"]
].to_pandas()
locations_df = locations_df.merge(
    median_values,
    left_on=["Naam", "Grootheid.Code"],
    right_on=["Naam", "Grootheid.Code"],
    how="inner",
)

locations_df = locations_df.dropna(subset=["median"])  # Drop rows without median values

scatter = ax.scatter(
    locations_df["Lon"],
    locations_df["Lat"],
    # c=locations_df["median"],
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
        f"{row['LocationCode']}\n{row['Grootheid.Code']}\n",  # med: {row['median']:.2f}
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
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.title("Wind Measurement Locations (OpenStreetMap)")
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\wind_locations_map.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Create interactive map with leaflet (folium)
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
    if color_val < 0.25:
        color = "blue"
    elif color_val < 0.5:
        color = "green"
    elif color_val < 0.75:
        color = "yellow"
    else:
        color = "red"

    popup_text = f"""
    <b>{row['LocationCode']}</b><br>
    Parameter: {row['Grootheid.Code']}<br>
    Median: {row['median']:.2f} {row['Eenheid.Code']}
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
     <p style="margin: 0;"><b>Wind Levels</b></p>
     <p style="margin: 5px 0;"><i style="background:blue;width:20px;height:20px;display:inline-block;"></i> Low</p>
     <p style="margin: 5px 0;"><i style="background:green;width:20px;height:20px;display:inline-block;"></i> Low-Mid</p>
     <p style="margin: 5px 0;"><i style="background:yellow;width:20px;height:20px;display:inline-block;"></i> Mid-High</p>
     <p style="margin: 5px 0;"><i style="background:red;width:20px;height:20px;display:inline-block;"></i> High</p>
</div>
"""
# Save interactive map
m.save(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\interactive_map_wind_locations2.html"
)
print("Interactive map saved to: interactive_map_wind_locations.html")
from IPython.display import display

display(m)
# %%
locations_df
# %%

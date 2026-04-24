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

ROOT_DIR = Path(r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse")

DATA_OUTPUT_DIR = ROOT_DIR / "data" / "4_output"
# %%

locations = ddlpy.locations()

## also add code as a variable
locations["Code"] = locations.index
locations.index.name = "LocationCode"
locations_pl = pl.from_pandas(locations, include_index=True)
locations_pl
# %%
locations_pl = locations_pl.filter(
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
    (pl.col("ProcesType") == "meting")
    & (pl.col("Grootheid.Code") == "WATHTE")
    & (pl.col("Hoedanigheid.Code") == "NAP")
    & (pl.col("Groepering.Code").is_in([""]))
    & (pl.col("WaardeBewerkingsMethode.Omschrijving").is_in(["NVT"]))
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
        # 'breezanddijk',
        "denoever.ijsselmeer.binnenhaven",
        "denoever.waddenzee.voorhaven",
        "dronten.roggebotsluis.drontermeer",
        "dronten.roggebotsluis.vossemeer",
        # 'enkhuizen',
        # 'enkhuizen.ijsselmeer.oost.meetpaal1',
        # 'enkhuizen.ijsselmeer.oost.meetpaal2',
        "genemuiden",
        # 'ijsselmeer.midden.meetpaal26',
        "kadoelen.zwanediep",
        "kampen.keteldiep",
        #  'kamperhoek.badstrand',
        "kampen.ijssel",
        "ketelhaven.ketelmeer",
        "kornwerderzand.ijsselmeer.binnenhaven",
        "kornwerderzand.waddenzee.buitenhaven",
        "krabbersgat.ijsselmeer",
        "krabbersgat.markermeer",
        "lelystad.houtribsluis.noord",
        "lelystad.houtribsluis.zuid",
        "lemmer.ijsselmeer",
        "lemmer.ijsselmeer.woudagemaal",
        "lemmer.woudagemaal.stroomkanaal",
        # 'markerwaarddijk.ijsselmeer.zuidwest.meetpaal37',
        "oudemirdum.ijsselmeer.noordoost.meetpaal9",
        # 'oudemirdum.ijsselmeer.noordoost.meetpaalfl05',
        "oudezeug.ijsselmeer.midden.noord.meetpaal47",
        "oudezeug.ijsselmeer.midden.noord.meetpaal48",
        "oudezeug.ijsselmeer.midden.noord.meetpaal49",
        # 'paardenboer',
        "ramspolbrug.ramsdiep",
        "ramspolkering.ketelmeer",
        "rotterdamsehoek.ijsselmeer",
        "rotterdamsehoek.meetpaalfl2",
        # 'schokkerhaven',
        "stavoren.gemaalhoogland.buiten"
        # 'stavoren.ijsselmeer',
        "swifterbant.ketelmeer",
        #  'urk',
        "urk.ijsselmeer.oost.meetpaal2",
        # 'urk.ijsselmeer.zuid.meetpaal46',
        "zwolle.ijssel",
    ]
)
bool_procestype = locations["ProcesType"].isin(["meting"])
bool_grootheid = locations["Grootheid.Code"].isin(["WATHTE"])
bool_groepering = locations["Groepering.Code"].isin([""])
bool_hoedanigheid = locations["Hoedanigheid.Code"].isin(["NAP"])
bool_bewerkingsmethode = locations["WaardeBewerkingsMethode.Omschrijving"].isin(["NVT"])

selected = locations.loc[
    bool_stations
    & bool_procestype
    & bool_grootheid
    & bool_groepering
    & bool_hoedanigheid
    & bool_bewerkingsmethode
]
selected
# %%
unique_names = selected["Naam"].unique()
unique_names
# %%


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
# %%
# EXPORT TO PARQUET

TODAY = dt.datetime.now().strftime("%Y%m%d")

measurements.write_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_waterhoogte_2018_2026_{TODAY}.parquet"
)
measurements.head()
# %%
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height

group_cols = [
    "Code",
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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\2_external\measurements_RWS_waterhoogte_per_combo"
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
measurements_daily.group_by("Naam").agg(pl.len()).sort("len", descending=True)
# %%
measurements_daily
# %%

unique_names = measurements_daily["Naam"].unique()
num_plots = len(unique_names)
fig, axes = plt.subplots(
    num_plots, 1, figsize=(12, 4 * num_plots), sharex=True, sharey=True
)

for idx, naam in enumerate(unique_names):
    filtered = measurements_daily.filter(pl.col("Naam") == naam)
    axes[idx].plot(
        filtered["time"], filtered["Meetwaarde.Waarde_Numeriek"], linewidth=0.5
    )
    axes[idx].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
    axes[idx].set_title(f"Measurements for {naam}")
    axes[idx].set_ylabel("Waarde_Numeriek")
    axes[idx].set_xlabel("")
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2026, 4, 1))
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\all_measurements_by_naam.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()
# %%
# Generate median, 90th percentile, and 10th percentile per Naam
percentile_stats = (
    measurements_daily.group_by(["Naam", "Grootheid.Code", "Eenheid.Code"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().round(2).alias("mean"),
        pl.col("Meetwaarde.Waarde_Numeriek").median().round(2).alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).round(2).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).round(2).alias("p10"),
    )
    .sort(["Naam", "Grootheid.Code", "Eenheid.Code"])
)
percentile_stats
# %%
# Plot showing 1 year with each year colored differently (using monthly means)

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
        axes[idx].axhline(y=0, color="red", linestyle="--", linewidth=0.5)

    axes[idx].set_xlabel("Month")
    axes[idx].set_ylabel("Monthly Median Meetwaarde")
    axes[idx].set_title(f"Monthly Median Meetwaarde by Month (1 year cycle) - {naam}")
    axes[idx].set_xticks(range(1, 13))
    axes[idx].legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\monthly_median_meetwaarde_by_month.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Plot showing 1 year with each year colored differently (using monthly means)

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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\monthly_median_meetwaarde_by_month_with_ci.png",
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
        f"{row['Naam']}\n{row['Grootheid.Code']}\nmed: {row['median']:.2f} {row['Eenheid.Code']}",
        fontsize=7,
        ha=ha,
        va=va,
        bbox=dict(
            boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="grey"
        ),
        zorder=6,
    )

# Add OpenStreetMap basemap
ax.set_xlim(4.85, 6.15)
ax.set_ylim(52.45, 53.1)
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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\waterhoogte_locations_map.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# Plot interactive map with folium

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
    Median: {row["median"]:.2f} {row["Eenheid.Code"]}
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
     <p style="margin: 0;"><b>Waterhoogte Levels</b></p>
     <p style="margin: 5px 0;"><i style="background:purple;width:20px;height:20px;display:inline-block;"></i> Low</p>
     <p style="margin: 5px 0;"><i style="background:blue;width:20px;height:20px;display:inline-block;"></i> Low-Mid</p>
     <p style="margin: 5px 0;"><i style="background:green;width:20px;height:20px;display:inline-block;"></i> Mid-High</p>
     <p style="margin: 5px 0;"><i style="background:yellow;width:20px;height:20px;display:inline-block;"></i> High</p>
</div>
"""
# Save interactive map
m.save(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\interactive_map_waterhoogte_locations.html"
)
print("Interactive map saved to: interactive_map_waterhoogte_locations.html")
# %%

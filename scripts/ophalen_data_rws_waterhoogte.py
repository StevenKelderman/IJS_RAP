# %%
# enabling debug logging so we can see what happens in the background
import datetime as dt
import logging
import re
from pathlib import Path

import ddlpy
import folium
import matplotlib.pyplot as plt
import polars as pl

logging.basicConfig()
logging.getLogger("ddlpy").setLevel(logging.INFO)

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
        (pl.col("LocationCode").str.contains("ijsselmeer"))
        | (pl.col("LocationCode").str.contains("denoever"))
        | (pl.col("LocationCode").str.contains("kornwerderzand"))
        | (pl.col("LocationCode").str.contains("krabbersgat"))
        | (pl.col("LocationCode").str.contains("houtrib"))
        | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
        | (pl.col("LocationCode").str.contains("kampen.ijssel"))
    )
    & (pl.col("ProcesType") == "meting")
    & (pl.col("Grootheid.Code") == "WATHTE")
    & (pl.col("Hoedanigheid.Code") == "NAP")
    & (pl.col("Groepering.Code").is_in([""]))
    & (pl.col("WaardeBewerkingsMethode.Omschrijving").is_in(["NVT"]))
)
# filter for water level stations
locations_pl
# %%
unique_location_codes = locations_pl["LocationCode"].unique().to_list()
unique_location_codes.sort()
print(unique_location_codes)
# %%

bool_stations = locations.index.isin(
    [
        "denoever.ijsselmeer.binnenhaven",
        "denoever.waddenzee.voorhaven",
        # 'enkhuizen.ijsselmeer.oost.meetpaal1',
        # 'enkhuizen.ijsselmeer.oost.meetpaal2',
        # 'ijsselmeer.midden.meetpaal26',
        "kampen.ijssel",
        "kornwerderzand.ijsselmeer.binnenhaven",
        "kornwerderzand.waddenzee.buitenhaven",
        "krabbersgat.ijsselmeer",
        "krabbersgat.markermeer",
        "lelystad.houtribsluis.noord",
        "lelystad.houtribsluis.zuid",
        "lemmer.ijsselmeer",
        "lemmer.ijsselmeer.woudagemaal",
        # 'markerwaarddijk.ijsselmeer.zuidwest.meetpaal37',
        "oudemirdum.ijsselmeer.noordoost.meetpaal9",
        # 'oudemirdum.ijsselmeer.noordoost.meetpaalfl05',
        "oudezeug.ijsselmeer.midden.noord.meetpaal47",
        "oudezeug.ijsselmeer.midden.noord.meetpaal48",
        "oudezeug.ijsselmeer.midden.noord.meetpaal49",
        "rotterdamsehoek.ijsselmeer",
        # 'stavoren.ijsselmeer',
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
# %%
# EXPORT TO PARQUET

TODAY = dt.datetime.now().strftime("%Y%m%d")

measurements.write_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_waterhoogte_2018_2026_{TODAY}.parquet"
)
measurements.shape

# %%
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height

group_cols = ["Naam", "Grootheid.Code", "Eenheid.Code"]

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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\monthly_median_meetwaarde_by_month_with_ci.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
# X-Y plot for Houtribsluis locations
import matplotlib.pyplot as plt
import numpy as np

houtribsluis_data = measurements.filter(pl.col("Naam").str.contains("Houtribsluis"))

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
    # plt.savefig(r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\waterhoogte\houtribsluis_comparison.png", dpi=300, bbox_inches="tight")
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
    <b>{row['Naam']}</b><br>
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
from IPython.display import display

display(m)
# %%

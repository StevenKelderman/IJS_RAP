# %%
# enabling debug logging so we can see what happens in the background
import datetime as dt
import logging

import ddlpy
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
        | (pl.col("LocationCode").str.contains("markermeer"))
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
print(unique_location_codes)
# %%

bool_stations = locations.index.isin([
    "enkhuizen.ijsselmeer.oost.meetpaal1",
    "markerwaarddijk.ijsselmeer.zuidwest.meetpaal37",
    "denoever.waddenzee.voorhaven",
    "krabbersgat.markermeer",
    "kornwerderzand.waddenzee.buitenhaven",
    "urk.ijsselmeer.oost.meetpaal2",
    "stavoren.ijsselmeer",
    "rotterdamsehoek.ijsselmeer",
    "oudezeug.ijsselmeer.midden.noord.meetpaal47",
    "krabbersgat.ijsselmeer",
    "markermeer.midden",
    "markermeer.trintelzand",
    "oudemirdum.ijsselmeer.noordoost.meetpaalfl05",
    "enkhuizen.ijsselmeer.oost.meetpaal2",
    "urk.ijsselmeer.zuid.meetpaal46",
    "kornwerderzand.ijsselmeer.binnenhaven",
    "lelystad.houtribsluis.noord",
    "lelystad.houtribsluis.zuid",
    "lemmer.ijsselmeer.woudagemaal",
    "denoever.ijsselmeer.binnenhaven",
    "oudemirdum.ijsselmeer.noordoost.meetpaal9",
    "oudezeug.ijsselmeer.midden.noord.meetpaal49",
    "lemmer.ijsselmeer",
    "oudezeug.ijsselmeer.midden.noord.meetpaal48",
    "markermeer.midden.meetpaal42",
    "ijsselmeer.midden.meetpaal26",
])
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

all_measurements = []
print(f"Starting processing of {len(selected)} stations...")
for idx, (index, row) in enumerate(selected.iterrows()):
    station_name = row["Naam"]
    print(f"({idx + 1}/{len(selected)}) Getting measurements for {station_name}")
    print("  Start date: 2018-01-01, End date: 2025-04-01")
    measurements = ddlpy.measurements(row, "2018-01-01", "2025-04-01")
    print(f"  Retrieved {len(measurements)} measurements for {station_name}")
    all_measurements.append(measurements)
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
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_waterhoogte_{TODAY}.parquet"
)
# %%
measurements.shape
# %%
measurements.group_by("Naam").agg(pl.count()).sort("count", descending=True)

# %%
measurements.plot(y="Meetwaarde.Waarde_Numeriek", linewidth=0.5, figsize=(12, 6))

# %%
measurements

# %%
# Generate median, 90th percentile, and 10th percentile per Naam
percentile_stats = (
    measurements
    .group_by(["Naam", "Grootheid.Code"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").median().alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).alias("p10"),
    )
    .sort(["Naam", "Grootheid.Code"])
)
percentile_stats

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

unique_names = measurements["Naam"].unique()

num_plots = len(unique_names)
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(unique_names):
    data = measurements.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    sns.lineplot(
        data=data_sorted.to_pandas(),
        x="time",
        y="Meetwaarde.Waarde_Numeriek",
        ax=axes[idx],
    )
    axes[idx].set_xlabel("Date")
    axes[idx].set_ylabel("Meetwaarde")
    axes[idx].set_title(f"Meetwaarde over Time - {naam}")
    axes[idx].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %%
# Plot showing 1 year with each year colored differently (using monthly means)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
colors = plt.cm.plasma(np.linspace(0.3, 1, 10))

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
        data_pd
        .groupby("year_month")
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
plt.show()

# %%
# Plot showing 1 year with each year colored differently (using monthly means)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
        data_pd
        .groupby("year_month")
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
plt.show()
# %%
locations_pl_coords = locations_pl.select([
    "LocationCode",
    "Lat",
    "Lon",
    "Naam",
    "Grootheid.Code",
]).unique()
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
median_values = percentile_stats[["Naam", "Grootheid.Code", "median"]].to_pandas()
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
        f"{row['Naam']}\n{row['Grootheid.Code']}\nmed: {row['median']:.2f}",
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
plt.title("Salinity Measurement Locations (OpenStreetMap)")
plt.show()

# %%

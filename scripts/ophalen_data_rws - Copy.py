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
        (
            (pl.col("LocationCode").str.contains("ijsselmeer"))
            | (pl.col("LocationCode").str.contains("denoever"))
            | (pl.col("LocationCode").str.contains("kornwerderzand"))
            | (pl.col("LocationCode").str.contains("andijk"))
        )
        & (pl.col("ProcesType") == "meting")
        & (pl.col("Hoedanigheid.Code") == "Cl")
    )
    | (
        (
            (pl.col("LocationCode").str.contains("ijsselmeer"))
            | (pl.col("LocationCode").str.contains("denoever"))
            | (pl.col("LocationCode").str.contains("kornwerderzand"))
            | (pl.col("LocationCode").str.contains("andijk"))
        )
        & (pl.col("ProcesType") == "meting")
        & (pl.col("Grootheid.Code") == "SALNTT")
    )
)  # filter for salinity stations
locations_pl
# %%
unique_location_codes = locations_pl["LocationCode"].unique().to_list()
print(unique_location_codes)


# %%

bool_stations = locations.index.isin([
    "andijk",
    "denoever.ijsselmeer.stevinsluizen.zuid",
    "denoever.ijsselmeer.vaargeul",
    "denoever.spuisluis.binnen",
    "kornwerderzand.ijsselmeer.spui",
    "kornwerderzand.ijsselmeer.binnenhaven",
    "kornwerderzand.ijsselmeer.vaargeul",
    "kornwerderzand.ijsselmeer.spuihavenzuid",
    "kornwerderzand.ijsselmeer.middelgronden",
    "denoever.waddenzee.voorhaven.kolk",
    "kornwerderzand.waddenzee.voorhaven.kolk",
    "friesekust.ijsselmeer",
    "wieringen.ijsselmeer.oost",
    "rotterdamsehoek.ijsselmeer",
    "denoever.ijsselmeer.binnenhaven",
])
bool_procestype = locations["ProcesType"].isin([
    "meting"
])  # meting/astronomisch/verwachting
bool_grootheid_CONC = locations["Grootheid.Code"].isin(["CONCTTE"])  # salinity (SALNTT)
bool_grootheid_SAL = locations["Grootheid.Code"].isin(["SALNTT"])  # salinity (SALNTT)
bool_hoedanigheid_CL = locations["Hoedanigheid.Code"].isin([
    "Cl"
])  # vertical reference (NAP/MSL)
bool_hoedanigheid_SAL = locations["Hoedanigheid.Code"].isin(["NVT"])
selected = locations.loc[
    bool_stations
    & bool_procestype
    & (bool_grootheid_CONC | bool_grootheid_SAL)
    & (bool_hoedanigheid_CL | bool_hoedanigheid_SAL)
]
selected

selected_pl = pl.from_pandas(selected, include_index=True)
selected_pl
# %%
unique_location_codes = selected_pl["LocationCode"].unique().to_list()
unique_location_codes
# %%

# if we pass one row to the measurements function you can get all the measurements
all_measurements = []
print(f"Starting processing of {len(selected)} stations...")
for idx, row in selected.iterrows():
    station_name = row["Naam"]
    print(f"Getting measurements for {station_name} (index {idx})")
    print("  Start date: 2018-01-01, End date: 2025-04-01")
    measurements = ddlpy.measurements(row, "2018-01-01", "2025-04-01")
    print(f"  Retrieved {len(measurements)} measurements for {station_name}")
    all_measurements.append(measurements)
print(f"Completed processing of all {len(selected)} stations")

# %%
non_empty_count = sum(1 for df in all_measurements if not df.empty)
print(f"Number of non-empty dataframes: {non_empty_count}")
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
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_saliniteit_{TODAY}.parquet"
)

# %%
# Read parquet file
measurements = pl.read_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_saliniteit_{TODAY}.parquet"
)

# %%
measurements.shape
# %%
measurements.group_by(["Naam", "Grootheid.Code"]).agg(pl.count()).sort(
    "count", descending=True
)

# %%
# measurements.plot(y="Meetwaarde.Waarde_Numeriek", linewidth=0.5, figsize=(12, 6))

# %%
measurements

# %%
# Generate median, 90th percentile, and 10th percentile per Naam
percentile_stats = (
    measurements
    .group_by("Naam")
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").median().alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).alias("p10"),
    )
    .sort("Naam")
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
    data_pd = data_sorted.to_pandas()

    # Calculate monthly median values
    data_pd["year_month"] = data_pd["time"].dt.to_period("M")
    monthly_median = data_pd.groupby("year_month")[
        "Meetwaarde.Waarde_Numeriek"
    ].median()
    monthly_median.index = monthly_median.index.to_timestamp()

    axes[idx].plot(
        monthly_median.index, monthly_median.values, marker="o", markersize=4
    )
    axes[idx].set_xlabel("Date")
    axes[idx].set_ylabel("Meetwaarde (Median)")
    axes[idx].set_title(f"Meetwaarde Median per Month - {naam}")
    axes[idx].tick_params(axis="x", rotation=45)
    axes[idx].set_xlim(dt.datetime(2018, 1, 1), dt.datetime(2025, 4, 1))

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
# Plot showing 1 subplot per year with each location plotted
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
colors = plt.cm.tab20(np.linspace(0, 1, 20))

measurements = measurements.with_columns(pl.col("time").dt.year().alias("year"))
measurements = measurements.with_columns(pl.col("time").dt.month().alias("month"))

# Get unique years and locations
unique_years = sorted(measurements["year"].unique().to_list())
unique_names = sorted(measurements["Naam"].unique().to_list())

num_plots = len(unique_years)
fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))

if num_plots == 1:
    axes = [axes]

for year_idx, year in enumerate(unique_years):
    year_data = measurements.filter(pl.col("year") == year)

    # Plot each location with different color
    for loc_idx, naam in enumerate(unique_names):
        location_data = year_data.filter(pl.col("Naam") == naam)

        if len(location_data) > 0:
            # Calculate monthly mean for each location
            monthly_mean = (
                location_data
                .group_by("month")
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
plt.show()

# %%
locations_pl_coords = locations_pl.select([
    "LocationCode",
    "Lat",
    "Lon",
    "Naam",
    "Grootheid.Code",
])

# %%
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

# Create figure with ERTS89 (EPSG:4258) projection
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()})

# Plot locations
locations_df = locations_pl_coords.to_pandas()
ax.scatter(
    locations_df["Lon"],
    locations_df["Lat"],
    transform=ccrs.PlateCarree(),
    s=100,
    alpha=0.6,
    edgecolors="k",
)

# Add labels for each location
for idx, row in locations_df.iterrows():
    ax.text(
        row["Lon"],
        row["Lat"],
        row["Naam"],
        transform=ccrs.PlateCarree(),
        fontsize=8,
        ha="right",
    )

# ax.add_feature(cfeature.COASTLINE, zorder=-20)
ax.add_feature(cfeature.LAKES, zorder=-50)
ax.add_feature(cfeature.OCEAN, zorder=-100)
ax.gridlines(draw_labels=True)
# Set extent to Netherlands (lon_min, lon_max, lat_min, lat_max)
ax.set_extent([4.7, 5.75, 52.5, 53.1], crs=ccrs.PlateCarree())
plt.title("Salinity Measurement Locations (ERTS89)")
plt.show()
# %%

# %%
# Imports
import datetime as dt
import logging
import re
from pathlib import Path

import ddlpy
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from contextily import add_basemap

# enabling debug logging so we can see what happens in the background
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
            | (pl.col("LocationCode").str.contains("waddenzee"))
            # | (pl.col("LocationCode").str.contains("markermeer"))
            # | (pl.col("LocationCode").str.contains("doovebalg"))
            | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
            | (pl.col("LocationCode").str.contains("kampen.ijssel"))
            | (pl.col("LocationCode").str.contains("ijssel.km"))
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
            | (pl.col("LocationCode").str.contains("waddenzee"))
            # | (pl.col("LocationCode").str.contains("markermeer"))
            # | (pl.col("LocationCode").str.contains("doovebalg"))
            | (pl.col("LocationCode").str.contains("zwolle.ijssel"))
            | (pl.col("LocationCode").str.contains("kampen.ijssel"))
            | (pl.col("LocationCode").str.contains("ijssel.km"))
        )
        & (pl.col("ProcesType") == "meting")
        & (pl.col("Grootheid.Code") == "SALNTT")
    )
)  # filter for salinity stations
locations_pl
# %%
unique_location_codes = locations_pl["LocationCode"].unique().sort().to_list()
print(unique_location_codes)
# %%

bool_stations = locations.index.isin([
    'andijk',
    'denoever.ijsselmeer.binnenhaven',
    'denoever.ijsselmeer.stevinsluizen.zuid',
    'denoever.ijsselmeer.vaargeul',
    'denoever.spuisluis.binnen',
    'denoever.waddenzee.voorhaven.kolk', 
    'friesekust.ijsselmeer', 
    'kornwerderzand.ijsselmeer.binnenhaven', 
    'kornwerderzand.ijsselmeer.middelgronden', 
    'kornwerderzand.ijsselmeer.spui', 
    'kornwerderzand.ijsselmeer.spuihavenzuid', 
    'kornwerderzand.ijsselmeer.vaargeul', 
    'kornwerderzand.waddenzee.voorhaven.kolk', 
    'rotterdamsehoek.ijsselmeer', 
    'waddenzee.ra', 
    'wieringen.ijsselmeer.oost'])

bool_procestype = locations["ProcesType"].isin(["meting"])
bool_grootheid_CONC = locations["Grootheid.Code"].isin(["CONCTTE"])  # salinity (SALNTT)
bool_grootheid_SAL = locations["Grootheid.Code"].isin(["SALNTT"])  # salinity (SALNTT)
bool_parameter_CL = locations["Parameter.Code"].isin(["Cl"])
bool_parameter_SAL = locations["Parameter.Code"].isin(["NVT"])

selected = locations.loc[
    bool_stations
    & bool_procestype
    & (
        (bool_grootheid_CONC & bool_parameter_CL)
        | (bool_grootheid_SAL & bool_parameter_SAL)
    )
]
selected

selected_pl = pl.from_pandas(selected, include_index=True)
selected_pl
# %%
unique_location_codes = selected_pl["LocationCode"].unique().to_list()
unique_location_codes

# %%
unique_location_codes = (
    selected_pl
    .select(["LocationCode", "Naam", "Omschrijving"])
    .unique()
    .sort("LocationCode")
)
unique_location_codes
# %%

# if we pass one row to the measurements function you can get all the measurements
all_measurements = []
print(f"Starting processing of {len(selected)} stations...")
for idx, (index, row) in enumerate(selected.sort_values("Naam").iterrows()):
    station_name = row["Naam"]
    grootheid = row["Grootheid.Code"]
    print(
        f"{idx + 1}/{len(selected)}: Getting measurements for {station_name} ({grootheid})"
    )
    print("  Start date: 2018-01-01, End date: 2026-03-31")
    measurements = ddlpy.measurements(row, "2018-01-01", "2026-03-31")
    print(
        f"  Retrieved {len(measurements):,} measurements for {station_name} ({grootheid})"
    )
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
# EXPORTED ON 21-04-2026 NO NEED FOR ANOTHER ONE
TODAY = dt.datetime.now().strftime("%Y%m%d")

measurements.write_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_saliniteit_2018_2026_{TODAY}.parquet"
)

# %%
# Read parquet file
TODAY = dt.datetime.now().strftime("%Y%m%d")
# TODAY = "20260401"  # override for testing

measurements = pl.read_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_saliniteit_2018_2026_{TODAY}.parquet"
)

# Convert bodemhoogte from string to int
measurements = measurements.with_columns(
    pl
    .col("WaarnemingMetadata.Bemonsteringshoogte")
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

measurements = measurements.filter(
    pl
    .col("WaarnemingMetadata.Kwaliteitswaardecode").is_in(["00", "25"])
    .alias("WaarnemingMetadata.Kwaliteitswaardecode")
)

measurements.shape
# %%
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height
hoogte_col = "WaarnemingMetadata.Bemonsteringshoogte"
if hoogte_col not in measurements.columns and "WaarnemingsHoogte" in measurements.columns:
    hoogte_col = "WaarnemingsHoogte"

group_cols = ["Naam", "Grootheid.Code", "Eenheid.Code", hoogte_col]

# Aggregate 10-minute data to daily means per station/parameter/height combination
measurements_daily = (
    measurements
    .with_columns(pl.col("time").cast(pl.Datetime, strict=False).dt.date().alias("time"))
    .group_by(group_cols + ["time"])
    .agg(pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("Meetwaarde.Waarde_Numeriek"))
    .sort(group_cols + ["time"])
)

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
measurements_daily.group_by([
    "Naam",
    "Grootheid.Code",
    "WaarnemingMetadata.Bemonsteringshoogte",
]).agg(pl.len()).sort("len", descending=True)

measurements_daily
# %%
# Completeness check: years as columns, count of Meetwaarde values per location
completeness = (
    measurements_daily
    .with_columns(
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
    measurements_daily
    .group_by([
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
    ])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("mean"),
        pl.col("Meetwaarde.Waarde_Numeriek").median().alias("median"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.90).alias("p90"),
        pl.col("Meetwaarde.Waarde_Numeriek").quantile(0.10).alias("p10"),
    )
    .sort([
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
    ])
)
percentile_stats
# %%
# Plot time series of Meetwaarde for each station
sns.set_style("whitegrid")

num_plots = len(measurements_daily["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements_daily["Naam"].unique().sort()):
    print(f"({idx + 1}/{num_plots}) Plotting daily mean time series for station: {naam}")
    data = measurements_daily.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()
    data_pd["WaarnemingMetadata.Bemonsteringshoogte"] = (
        data_pd["WaarnemingMetadata.Bemonsteringshoogte"].astype(str)
    )
    sns.lineplot(
        data=data_pd,
        x="time",
        y="Meetwaarde.Waarde_Numeriek",
        hue="WaarnemingMetadata.Bemonsteringshoogte",
        ax=axes[idx],
    )
    axes[idx].set_xlabel("")
    axes[idx].set_ylabel("Meetwaarde (daily mean)")
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2026, 4, 1))
    axes[idx].set_title(f"Daily Mean Meetwaarde over Time - {naam}")
    axes[idx].tick_params(axis="x")
    axes[idx].legend(title="Bemonsteringshoogte")

plt.tight_layout()
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\time_series_daily_mean_meetwaarde_per_station.png",
             dpi=300
)
plt.show()

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
        seasonal_data
        .groupby(["year", "season", hoogte_col])["Meetwaarde.Waarde_Numeriek"]
        .median()
        .reset_index()
    )

    if seasonal_median.empty:
        axes[idx].set_title(f"Seasonal Median by Bemonsteringshoogte - {naam} (no data)")
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
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\seasonal_median_meetwaarde_per_station.png",
             dpi=300
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
        data_pd
        .groupby(["year_month", hoogte_col])["Meetwaarde.Waarde_Numeriek"]
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
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\monthly_median_meetwaarde_per_station.png",
             dpi=300
)
plt.show()

# %%
import pandas as pd

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
    measurements_a
    .select([
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        # "WaarnemingMetadata.Bemonsteringshoogte",
    ])
    .unique()
    .sort([
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        # "WaarnemingMetadata.Bemonsteringshoogte",
    ])
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
        data_pd
        .groupby(["year_month", "WaarnemingMetadata.Bemonsteringshoogte"])[
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
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\monthly_median_meetwaarde_boxplot_per_station.png", dpi=300)
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
    axes[idx].set_xlim(1, 12)
    axes[idx].legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\monthly_median_meetwaarde_by_month_yearly_colors_per_station.png", dpi=300)
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
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\monthly_median_meetwaarde_by_month_yearly_colors_ci_per_station.png", dpi=300)
plt.show()
# %%
# Plot showing 1 subplot per year with each location plotted
sns.set_style("whitegrid")

colors = plt.cm.tab20(np.linspace(0, 1, 20))

measurements_daily = measurements_daily.with_columns(pl.col("time").dt.year().alias("year"))
measurements_daily = measurements_daily.with_columns(pl.col("time").dt.month().alias("month"))

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
plt.savefig("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\4_output\\figuren\\monthly_mean_meetwaarde_by_location_yearly_subplots.png", dpi=300)
plt.show()
# %%
# X-Y plot for Houtribsluis locations

combs = [
    #  ("Kornwerderzand IJsselmeer spui", "Kornwerderzand Spuihaven zuid"),
    ("Den Oever Stevinsluizen zuid", "Den Oever, spuisluis, binnen"),
]

# Create list of all names from combs pairs
all_names = [name for pair in combs for name in pair]

houtribsluis_data = measurements.filter(pl.col("Naam").is_in(all_names))

if len(houtribsluis_data) > 0:
    houtribsluis_df = houtribsluis_data.sort("time").to_pandas()

    fig, ax = plt.subplots(figsize=(14, 6))

    unique_names = houtribsluis_df["Naam"].unique()
    if len(unique_names) >= 2:
        data1 = houtribsluis_df[houtribsluis_df["Naam"] == unique_names[0]].reset_index(
            drop=True
        )
        data2 = houtribsluis_df[houtribsluis_df["Naam"] == unique_names[1]].reset_index(
            drop=True
        )
        # Align data to same length
        min_len = min(len(data1), len(data2))
        data1 = data1.iloc[:min_len].copy()
        data2 = data2.iloc[:min_len].copy()
        data1["year"] = data1["time"].dt.year
        colors = data1["year"].apply(lambda x: plt.cm.Set2(x % 10))
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
        # Add 1:10 line
        x_range = np.array([
            data1["Meetwaarde.Waarde_Numeriek"].min(),
            data1["Meetwaarde.Waarde_Numeriek"].max(),
        ])
        ax.plot(x_range / 10, x_range, "g--", linewidth=2, label="1:10 line")
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
    ax.set_ylim(0, 11000)  # Set y-axis to start at 0
    ax.set_xlim(0, 500)  # Set x-axis to start at 0
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
else:
    print("No Houtribsluis data found")

# %%
locations_pl_coords = locations_pl.select([
    "LocationCode",
    "Lat",
    "Lon",
    "Naam",
    "Grootheid.Code",
    "Eenheid.Code",
]).unique()

locations_pl_coords

# %%

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
locations_df = locations_df.loc[locations_df["Grootheid.Code"] == "CONCTTE"]
locations_df
# %%

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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\boxplot_meetwaarde_by_height_parameter_per_location.png",
    dpi=300
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
    
    param_data = measurements_daily.filter(pl.col("Grootheid.Code") == param).to_pandas()
    
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
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\boxplot_per_parameter_all_locations_heights.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
# %%
# Create interactive map with leaflet (folium)
# Calculate center of map
center_lat = locations_df["Lat"].mean()
center_lon = locations_df["Lon"].mean()

# Create folium map
m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=10,
    tiles="OpenStreetMap"
)

# Normalize median values for color mapping
min_median = locations_df["median"].min()
max_median = locations_df["median"].max()
locations_df["median_normalized"] = (locations_df["median"] - min_median) / (max_median - min_median)

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
    Median: {row['median']:.2f} {row['Eenheid.Code']}<br>
    Height: {row['WaarnemingMetadata.Bemonsteringshoogte']}m
    """
    
    folium.CircleMarker(
        location=[row["Lat"], row["Lon"]],
        radius=10,
        popup=folium.Popup(popup_text, max_width=300),
        color=color,
        fill=True,
        fillColor=color,
        fillOpacity=0.7,
        weight=2
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
m.save(r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\interactive_map_salinity_locations.html")
print("Interactive map saved to: interactive_map_salinity_locations.html")
from IPython.display import display

display(m)
# %%

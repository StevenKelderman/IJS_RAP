# %%
# Imports
import datetime as dt
import logging

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
            | (pl.col("LocationCode").str.contains("markermeer"))
            | (pl.col("LocationCode").str.contains("doovebalg"))
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
            | (pl.col("LocationCode").str.contains("markermeer"))
            | (pl.col("LocationCode").str.contains("doovebalg"))
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
    "andijk",  # -1.5
    "denoever.ijsselmeer.stevinsluizen.zuid",
    "denoever.ijsselmeer.vaargeul",
    "denoever.spuisluis.binnen",
    "denoever.ijsselmeer.binnenhaven",  # -1.25, -4
    "denoever.waddenzee.voorhaven.kolk",  # -2, -4
    "kornwerderzand.ijsselmeer.spui",
    "kornwerderzand.ijsselmeer.binnenhaven",  # -1.25, -4
    "kornwerderzand.ijsselmeer.vaargeul",
    "kornwerderzand.ijsselmeer.spuihavenzuid",
    "kornwerderzand.ijsselmeer.middelgronden",
    "kornwerderzand.waddenzee.voorhaven.kolk",  # -2, -4
    # "friesekust.ijsselmeer",
    # "wieringen.ijsselmeer.oost", # -1.5
    # "rotterdamsehoek.ijsselmeer",
    "markermeer.trintelzand",
    "markermeer.midden",
    # "waddenzee.ra",
    "doovebalg.oost",
    "doovebalg.west",
])
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
    print("  Start date: 2018-01-01, End date: 2026-04-01")
    measurements = ddlpy.measurements(row, "2018-01-01", "2026-04-01")
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

TODAY = dt.datetime.now().strftime("%Y%m%d")

measurements.write_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_saliniteit_{TODAY}.parquet"
)

# %%
# Read parquet file
TODAY = dt.datetime.now().strftime("%Y%m%d")
# TODAY = "20260401"  # override for testing

measurements = pl.read_parquet(
    rf"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\1_raw\measurements_RWS_saliniteit_{TODAY}.parquet"
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

# %%
measurements.shape
# %%
measurements.group_by([
    "Naam",
    "Grootheid.Code",
    "WaarnemingMetadata.Bemonsteringshoogte",
]).agg(pl.count()).sort("count", descending=True)
# %%
measurements

# %%
# Completeness check: years as columns, count of Meetwaarde values per location
completeness = (
    measurements
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
    measurements
    .group_by([
        "Naam",
        "Grootheid.Code",
        "Eenheid.Code",
        "WaarnemingMetadata.Bemonsteringshoogte",
    ])
    .agg(
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

num_plots = len(measurements["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements["Naam"].unique().sort()):
    print(f"({idx + 1}/{num_plots}) Plotting time series for station: {naam}")
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
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2026, 4, 1))
    axes[idx].set_title(f"Meetwaarde over Time - {naam}")
    axes[idx].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

# %%

# Plot seasonal median values (summer vs winter) for each station
sns.set_style("whitegrid")

num_plots = len(measurements["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting seasonal median time series for station: {naam}"
    )
    data = measurements.filter(pl.col("Naam") == naam)
    data_sorted = data.sort("time")
    data_pd = data_sorted.to_pandas()

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
        .groupby(["year", "season"])["Meetwaarde.Waarde_Numeriek"]
        .median()
        .reset_index()
    )

    # Plot summer and winter separately
    summer_data = seasonal_median[seasonal_median["season"] == "Summer"]
    winter_data = seasonal_median[seasonal_median["season"] == "Winter"]

    axes[idx].plot(
        summer_data["year"],
        summer_data["Meetwaarde.Waarde_Numeriek"],
        label="Summer",
        marker="o",
        linewidth=1.5,
        markersize=6,
    )
    axes[idx].plot(
        winter_data["year"],
        winter_data["Meetwaarde.Waarde_Numeriek"],
        label="Winter",
        marker="s",
        linewidth=1.5,
        markersize=6,
    )

    axes[idx].set_xlabel("Year")
    axes[idx].set_ylabel("Meetwaarde (Median)")
    axes[idx].set_title(f"Seasonal Median - {naam}")
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2025, 4, 1))
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%

# Plot median values for each station using monthly medians to smooth the data
sns.set_style("whitegrid")

num_plots = len(measurements["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting monthly median time series for station: {naam}"
    )
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
        monthly_median.index,
        monthly_median.values,
        marker="o",
        markersize=4,
        linewidth=1.5,
    )
    axes[idx].set_xlabel("Date")
    axes[idx].set_ylabel("Meetwaarde (Median)")
    axes[idx].set_title(f"Meetwaarde Median per Month - {naam}")
    axes[idx].tick_params(axis="x", rotation=45)
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2025, 4, 1))

plt.tight_layout()
plt.show()

# %%
import pandas as pd

measurements_a = measurements.with_columns(
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
        & (pl.col("WaarnemingMetadata.Kwaliteitswaardecode").is_in(["00", "03"]))
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
plt.show()
# %%
# Plot showing 1 year with each year colored differently (using monthly means)
sns.set_style("whitegrid")

colors = plt.cm.plasma(np.linspace(0.3, 1, 10))

num_plots = len(measurements["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting monthly median time series with yearly colors for station: {naam}"
    )
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
    axes[idx].set_xlim(dt.datetime(2017, 12, 31), dt.datetime(2025, 4, 1))
    axes[idx].legend(title="Year", bbox_to_anchor=(1.05, 1), loc="upper left")
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Plot showing 1 year with each year colored differently (using monthly means)
sns.set_style("whitegrid")

colors = plt.cm.plasma(np.linspace(0.3, 1, 10))

num_plots = len(measurements["Naam"].unique())
fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))

if num_plots == 1:
    axes = [axes]

for idx, naam in enumerate(measurements["Naam"].unique().sort()):
    print(
        f"({idx + 1}/{num_plots}) Plotting monthly median time series with yearly colors and confidence intervals for station: {naam}"
    )
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
ax.set_ylim(52.35, 53.2)
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

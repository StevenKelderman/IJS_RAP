# %%
# Imports
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# %%

PROJECT_DIR = Path(r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer")

DATA_OUTPUT_DIR = PROJECT_DIR / "analyse" / "data" / "4_output"

TODAY = dt.datetime.now().strftime("%Y%m%d")
# %%
# READ DEBIET CARLIJN
measurements_debiet_carlijn_daily = (
    pl.scan_parquet("P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\3_processed\\debieten_carlijn_2018_2022.parquet")
).collect()

measurements_debiet_carlijn_daily = measurements_debiet_carlijn_daily.rename(
    {"DATUM": "time",
    "WAARDE": "Meetwaarde.Waarde_Numeriek",
    "NAME": "Naam",})


measurements_debiet_carlijn_daily
# %%
# READ WADAR
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
measurements_debiet_daily = (
    pl.scan_parquet(PROJECT_DIR / "analyse" / "data" / "1_raw" / "per_location_per_year" / "debiet" / "*.parquet", missing_columns="insert")
    .filter(pl.col("WaarnemingMetadata.Kwaliteitswaardecode").cast(pl.Int32) <= 50)
    .with_columns(
        pl.col("time").cast(pl.Datetime, strict=False).dt.date().alias("time")
    )
    .group_by(group_cols + ["time"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("Meetwaarde.Waarde_Numeriek")
    )
    .sort(group_cols + ["time"])
    .collect()
)
measurements_debiet_daily

# %%
# READ WADAR
hoogte_col = "WaarnemingMetadata.Bemonsteringshoogte"

group_cols = ["Naam", "Grootheid.Code", "Eenheid.Code", hoogte_col]

# Aggregate 10-minute data to daily means per station/parameter/height combination
measurements_saliniteit_daily = (
    pl.scan_parquet(PROJECT_DIR / "analyse" / "data" / "1_raw" / "per_location_per_year" / "saliniteit" / "*.parquet", missing_columns="insert")
    .filter(pl.col("WaarnemingMetadata.Kwaliteitswaardecode").cast(pl.Int32) <= 50)
    .with_columns(
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
    .filter(
        pl.col("WaarnemingMetadata.Kwaliteitswaardecode")
        .is_in(["00", "25"])
        .alias("WaarnemingMetadata.Kwaliteitswaardecode")
    )
    .with_columns(
        pl.col("time").cast(pl.Datetime, strict=False).dt.date().alias("time")
    )
    .group_by(group_cols + ["time"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("Meetwaarde.Waarde_Numeriek")
    )
    .sort(group_cols + ["time"])
    .collect()
)

measurements_saliniteit_daily
# %%

group_cols = [
    "Naam",
    "Grootheid.Code",
    "Parameter_Wat_Omschrijving",
    "Eenheid.Code",
    "WaardeBewerkingsMethode.Omschrijving",
]
# READ WADAR and aggregate 10-minute data to daily means per station/parameter/height combination
measurements_wind_daily = (
    pl.scan_parquet(PROJECT_DIR / "analyse" / "data" / "1_raw" / "per_location_per_year" / "wind" / "*.parquet", missing_columns="insert")
    .filter(pl.col("WaarnemingMetadata.Kwaliteitswaardecode").cast(pl.Int32) <= 50)
    .collect()
    .with_columns(
        pl.col("time").cast(pl.Datetime, strict=False).dt.date().alias("time")
    )
    .group_by(group_cols + ["time"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("Meetwaarde.Waarde_Numeriek")
    )
    .sort(group_cols + ["time"])
)
measurements_wind_daily

# %%
# READ WADAR
# Export one CSV per unique combination of Naam, Grootheid.Code, Eenheid.Code and sampling height

group_cols = [
    "Code",
    "Naam",
    "Grootheid.Code",
    "Parameter_Wat_Omschrijving",
    "Eenheid.Code",
    "WaardeBewerkingsMethode.Omschrijving",
]

# Aggregate 10-minute data to daily means per station/parameter/height combination
measurements_waterhoogte_daily = (
    pl.scan_parquet(PROJECT_DIR / "analyse" / "data" / "1_raw" / "per_location_per_year" / "waterhoogte" / "*.parquet", missing_columns="insert")
    .filter(pl.col("WaarnemingMetadata.Kwaliteitswaardecode").cast(pl.Int32) <= 50)
    .with_columns(
        pl.col("time").cast(pl.Datetime, strict=False).dt.date().alias("time")
    )
    .group_by(group_cols + ["time"])
    .agg(
        pl.col("Meetwaarde.Waarde_Numeriek").mean().alias("Meetwaarde.Waarde_Numeriek")
    )
    .sort(group_cols + ["time"])
    .collect()
)
measurements_waterhoogte_daily
# %%
measurements_debiet_daily
# measurements_debiet_carlijn_daily
# measurements_saliniteit_daily
# measurements_waterhoogte_daily
# measurements_wind_daily

# %%
# %%
# Plot three parameters (debiet, wind, saliniteit) for 1 location and multiple years

# Select locations and years
LOCATION_DEBIET = "Olst"

LOCATION_DEBIET_CARLIJN = "Lorentz spuisluizen (Afsluitdijk)" # "Lorentz spuisluizen (Afsluitdijk)"; "Stevin spuisluizen (afsluitdijk)"
LOCATION_SALINITEIT = "Kornwerderzand Spuihaven zuid" # "Kornwerderzand Spuihaven zuid"; "Den Oever Stevinsluizen zuid"
LOCATION_WATERHOOGTE = "Kornwerderzand, IJsselmeer binnenhaven"

# LOCATION_DEBIET_CARLIJN = "Stevin spuisluizen (afsluitdijk)" # "Lorentz spuisluizen (Afsluitdijk)"; "Stevin spuisluizen (afsluitdijk)"
# LOCATION_SALINITEIT = "Den Oever Stevinsluizen zuid" # "Kornwerderzand Spuihaven zuid"; "Den Oever Stevinsluizen zuid"
# LOCATION_WATERHOOGTE = "Den Oever, IJsselmeer binnenhaven"

LOCATION_WIND = "Stavoren IJsselmeer"
YEARS = [2018, 2019, 2020, 2021, 2022]

# Define compass bins and labels (used for wind plots)
compass_directions = {
    "N": (337.5, 22.5),
    "NE": (22.5, 67.5),
    "E": (67.5, 112.5),
    "SE": (112.5, 157.5),
    "S": (157.5, 202.5),
    "SW": (202.5, 247.5),
    "W": (247.5, 292.5),
    "NW": (292.5, 337.5),
}

def get_wind_direction_bin(angle):
    """Bin wind direction angle into compass direction"""
    for direction, (min_angle, max_angle) in compass_directions.items():
        if min_angle <= max_angle:
            if min_angle <= angle < max_angle:
                return direction
        else:  # Handle wrap-around for North
            if angle >= min_angle or angle < max_angle:
                return direction
    return "N"

from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

for YEAR in YEARS:
    # Filter data for selected station and year
    debiet_data = measurements_debiet_daily.filter(
        (pl.col("Naam") == LOCATION_DEBIET) &
        (pl.col("time").dt.year() == YEAR)
    ).sort("time").to_pandas()

    debiet_carlijn_data = measurements_debiet_carlijn_daily.filter(
        (pl.col("Naam") == LOCATION_DEBIET_CARLIJN) &
        (pl.col("time").dt.year() == YEAR)
    ).sort("time").to_pandas()

    wind_data = measurements_wind_daily.filter(
        (pl.col("Naam") == LOCATION_WIND) &
        (pl.col("time").dt.year() == YEAR)
    ).sort("time")

    waterhoogte_data = measurements_waterhoogte_daily.filter(
        (pl.col("Naam") == LOCATION_WATERHOOGTE) &
        (pl.col("time").dt.year() == YEAR)
    ).sort("time")

    saliniteit_data = measurements_saliniteit_daily.filter(
        (pl.col("Naam") == LOCATION_SALINITEIT) &
        (pl.col("time").dt.year() == YEAR)
    ).sort("time").to_pandas()

    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(16, 10), sharex=True)

    # Plot 1: Debiet
    if len(debiet_data) > 0:
        axes[0].plot(debiet_data["time"], debiet_data["Meetwaarde.Waarde_Numeriek"], color="blue", linewidth=1)
        axes[0].set_ylabel("Debiet [m³/s]", fontsize=12)
        axes[0].set_title(f"Debiet - {LOCATION_DEBIET} ({YEAR})", fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, None)

    if len(debiet_carlijn_data) > 0:
        axes[1].plot(debiet_carlijn_data["time"], debiet_carlijn_data["Meetwaarde.Waarde_Numeriek"], color="blue", linewidth=1)
        axes[1].set_ylabel("Debiet [m³/s]", fontsize=12)
        axes[1].set_title(f"Debiet - {LOCATION_DEBIET_CARLIJN} ({YEAR})", fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, None)

    # Plot 2: Wind
    # Get wind direction and speed data
    wind_direction = wind_data.filter(
        pl.col("Grootheid.Code") == "WINDRTG"
    ).select(["Naam", "time", "Meetwaarde.Waarde_Numeriek"]).rename(
        {"Meetwaarde.Waarde_Numeriek": "wind_direction"}
    )

    wind_speed = wind_data.filter(
        pl.col("Grootheid.Code") == "WINDSHD"
    ).select(["Naam", "time", "Meetwaarde.Waarde_Numeriek"]).rename(
        {"Meetwaarde.Waarde_Numeriek": "wind_speed"}
    )

    # Join wind direction and speed data
    wind_combined = wind_speed.join(wind_direction, on=["Naam", "time"], how="inner")

    cbar = None
    if len(wind_combined) > 0:
        wind_combined_pandas = wind_combined.sort("time").to_pandas()

        # Bin wind directions
        wind_combined_pandas["wind_direction_bin"] = wind_combined_pandas["wind_direction"].apply(get_wind_direction_bin)

        # Map direction bins to numeric values for coloring
        direction_to_num = {direction: i for i, direction in enumerate(compass_directions.keys())}
        wind_combined_pandas["wind_direction_num"] = wind_combined_pandas["wind_direction_bin"].map(direction_to_num)

        # Create color map for 8 compass directions with rainbow colors
        cmap = plt.cm.get_cmap("hsv", 8)

        axes[2].plot(
            wind_combined_pandas["time"],
            wind_combined_pandas["wind_speed"],
            color="grey",
            linewidth=0.5,
            alpha=0.8,
            zorder=1
        )
        scatter = axes[2].scatter(
            wind_combined_pandas["time"],
            wind_combined_pandas["wind_speed"],
            c=wind_combined_pandas["wind_direction_num"],
            cmap=cmap,
            vmin=0,
            vmax=7,
            s=20,
            alpha=0.95,
            edgecolors="none",
            zorder=2
        )
        axes[2].set_ylabel("Wind snelheid [m/s]", fontsize=12)
        axes[2].set_ylim(0, wind_combined_pandas["wind_speed"].max() * 1.2)
        axes[2].set_title(f"Wind Speed with Direction Coloring - {LOCATION_WIND} ({YEAR})", fontsize=12)
        axes[2].grid(True, alpha=0.3)

        cax = inset_axes(axes[2], width="2%", height="80%", loc="center right", bbox_to_anchor=(0.01, 0, 1, 1), bbox_transform=axes[2].transAxes, borderpad=0)
        cbar = plt.colorbar(scatter, cax=cax, ticks=range(8))
        cbar.set_ticklabels(list(compass_directions.keys()))
        cbar.set_label("Wind Direction", rotation=270, labelpad=20)

    # Plot 3: Waterhoogte
    if len(waterhoogte_data) > 0:
        axes[3].plot(waterhoogte_data["time"], waterhoogte_data["Meetwaarde.Waarde_Numeriek"], color="blue", linewidth=1)
        axes[3].axhline(y=0, color="gray", linestyle="--", linewidth=1)
        axes[3].set_ylabel("Waterhoogte [cm NAP]", fontsize=12)
        axes[3].set_title(f"Waterhoogte - {LOCATION_WATERHOOGTE} ({YEAR})", fontsize=12)
        axes[3].grid(True, alpha=0.3)

    # Plot 4: Saliniteit
    if len(saliniteit_data) > 0:
        unique_heights = sorted(saliniteit_data["WaarnemingMetadata.Bemonsteringshoogte"].unique())
        cmap = plt.cm.get_cmap("tab10", len(unique_heights))
        color_map = {height: cmap(i) for i, height in enumerate(unique_heights)}
        colors = [color_map[height] for height in saliniteit_data["WaarnemingMetadata.Bemonsteringshoogte"]]
        scatter = axes[4].scatter(saliniteit_data["time"], saliniteit_data["Meetwaarde.Waarde_Numeriek"], c=colors, s=20, linewidth=0)
        axes[4].set_ylabel("Saliniteit [mg/l]", fontsize=12)
        axes[4].set_xlabel("Tijd", fontsize=12)
        axes[4].set_title(f"Saliniteit - {LOCATION_SALINITEIT} ({YEAR})", fontsize=12)
        axes[4].grid(True, alpha=0.3)

        # Add legend with height categories
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[height], markersize=8, label=f"{height} m") for height in unique_heights]
        axes[4].legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(
        DATA_OUTPUT_DIR / "figuren" / f"combined_parameters_{YEAR}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

# %%
# Wind speed plot with wind direction coloring (0-360 continuous)

measurements_wind_daily_stavoren = measurements_wind_daily.filter(pl.col("Naam") == "Stavoren IJsselmeer").sort("time")
# measurements_wind_daily_stavoren = measurements_wind_daily_stavoren.filter(pl.col("time").is_between(dt.datetime(2018, 1, 1), dt.datetime(2018, 12, 31)))
measurements_wind_daily_stavoren
# %%

# Get wind direction and speed data
wind_direction = measurements_wind_daily_stavoren.filter(
    pl.col("Grootheid.Code") == "WINDRTG"
).select(["Naam", "time", "Meetwaarde.Waarde_Numeriek"]).rename(
    {"Meetwaarde.Waarde_Numeriek": "wind_direction"}
)

wind_speed = measurements_wind_daily_stavoren.filter(
    pl.col("Grootheid.Code") == "WINDSHD"
).select(["Naam", "time", "Meetwaarde.Waarde_Numeriek"]).rename(
    {"Meetwaarde.Waarde_Numeriek": "wind_speed"}
)

# Join wind direction and speed data
wind_combined = wind_speed.join(wind_direction, on=["Naam", "time"], how="inner")

# Get unique station names
unique_stations = wind_combined.select("Naam").unique().sort("Naam")["Naam"].to_list()

# Define compass bins and labels
compass_directions = {
    "N": (337.5, 22.5),
    "NE": (22.5, 67.5),
    "E": (67.5, 112.5),
    "SE": (112.5, 157.5),
    "S": (157.5, 202.5),
    "SW": (202.5, 247.5),
    "W": (247.5, 292.5),
    "NW": (292.5, 337.5),
}

def get_wind_direction_bin(angle):
    """Bin wind direction angle into compass direction"""
    for direction, (min_angle, max_angle) in compass_directions.items():
        if min_angle <= max_angle:
            if min_angle <= angle < max_angle:
                return direction
        else:  # Handle wrap-around for North
            if angle >= min_angle or angle < max_angle:
                return direction
    return "N"

# Create subplots for each station
fig, axes = plt.subplots(len(unique_stations), 1, figsize=(14, 4 * len(unique_stations)))
if len(unique_stations) == 1:
    axes = [axes]

# Create color map for 8 compass directions with rainbow colors
colors = plt.cm.rainbow(np.linspace(0, 1, 8))
direction_colors = {direction: colors[i] for i, direction in enumerate(compass_directions.keys())}

for idx, station in enumerate(unique_stations):
    station_data = wind_combined.filter(pl.col("Naam") == station).sort("time").to_pandas()
    
    # Bin wind directions
    station_data["wind_direction_bin"] = station_data["wind_direction"].apply(get_wind_direction_bin)
    
    # Map direction bins to numeric values for coloring
    direction_to_num = {direction: i for i, direction in enumerate(compass_directions.keys())}
    station_data["wind_direction_num"] = station_data["wind_direction_bin"].map(direction_to_num)
    
    ax = axes[idx]
    ax.plot(
        station_data["time"],
        station_data["wind_speed"],
        color="grey",
        linewidth=0.5,
        alpha=0.8,
        zorder=1
    )
    scatter = ax.scatter(
        station_data["time"],
        station_data["wind_speed"],
        c=station_data["wind_direction_num"],
        cmap="rainbow",
        vmin=0,
        vmax=7,
        s=20,
        alpha=0.95,
        edgecolors="none",
        zorder=2

    )
    ax.set_xlabel("")
    ax.set_ylabel("Wind snelheid [m/s]", fontsize=12)
    ax.set_ylim(0, station_data["wind_speed"].max() * 1.2)
    # ax.set_xlim(dt.datetime(2018, 1, 1, 0, 0, 0), dt.datetime(2018, 1, 1, 23, 59, 59))
    ax.set_title(f"Wind Speed with Direction Coloring - {station}")
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(8))
    cbar.set_ticklabels(list(compass_directions.keys()))
    cbar.set_label("Wind Direction", rotation=270, labelpad=20)

plt.tight_layout()
plt.savefig(
    r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer\analyse\data\4_output\figuren\wind\wind_speed_with_direction_coloring.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

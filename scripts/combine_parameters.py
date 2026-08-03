# %%
# Imports
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# %%

PROJECT_DIR = Path(r"P:\11212806-wbb02-2026\009_Jaarrapportage_IJsselmeer")

DATA_OUTPUT_DIR = PROJECT_DIR / "analyse" / "data" / "4_output"

TODAY = dt.datetime.now().strftime("%Y%m%d")
# %%
# READ DEBIET CARLIJN
measurements_debiet_carlijn_daily = (
    pl.scan_parquet(
        "P:\\11212806-wbb02-2026\\009_Jaarrapportage_IJsselmeer\\analyse\\data\\3_processed\\debieten_carlijn_2018_2022.parquet"
    )
).collect()

measurements_debiet_carlijn_daily = measurements_debiet_carlijn_daily.rename(
    {
        "DATUM": "time",
        "WAARDE": "Meetwaarde.Waarde_Numeriek",
        "NAME": "Naam",
    }
)

measurements_debiet_carlijn_daily
# %%
# READ WADAR DEBIET
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
# READ WADAR SALINITEIT
hoogte_col = "WaarnemingMetadata.Bemonsteringshoogte"

group_cols = ["Naam", "Grootheid.Code", "Eenheid.Code", hoogte_col]

# Aggregate 10-minute data to daily means per station/parameter/height combination
measurements_saliniteit_daily = (
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
# READ WADAR WIND

group_cols = [
    "Naam",
    "Grootheid.Code",
    "Parameter_Wat_Omschrijving",
    "Eenheid.Code",
    "WaardeBewerkingsMethode.Omschrijving",
]
# READ WADAR and aggregate 10-minute data to daily means per station/parameter/height combination
measurements_wind_daily = (
    pl.scan_parquet(
        PROJECT_DIR
        / "analyse"
        / "data"
        / "1_raw"
        / "per_location_per_year"
        / "wind"
        / "*.parquet",
        missing_columns="insert",
    )
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

# Rename Rotterdames Hoek, meetpaal FL2 to Rotterdamsehoek
measurements_wind_daily = measurements_wind_daily.with_columns(
    pl.when(pl.col("Naam") == "Rotterdamse Hoek, meetpaal FL2")
    .then(pl.lit("Rotterdamsehoek IJsselmeer"))
    .otherwise(pl.col("Naam"))
    .alias("Naam")
)

measurements_wind_daily
# %%
# READ WADAR WATERHOOGTE
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
    pl.scan_parquet(
        PROJECT_DIR
        / "analyse"
        / "data"
        / "1_raw"
        / "per_location_per_year"
        / "waterhoogte"
        / "*.parquet",
        missing_columns="insert",
    )
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
# measurements_debiet_daily # m3/d / m3/s
# measurements_debiet_carlijn_daily
# measurements_saliniteit_daily # mg/l
# measurements_waterhoogte_daily # cm
# measurements_wind_daily # m/s
# %%
# Plot three parameters (debiet, wind, saliniteit) for 1 location and multiple years

# Select locations and years
LOCATION_DEBIET = "Olst"

LOCATION_DEBIET_CARLIJN = "Lorentz spuisluizen (Afsluitdijk)"  # "Lorentz spuisluizen (Afsluitdijk)"; "Stevin spuisluizen (afsluitdijk)"
LOCATION_SALINITEIT = "Kornwerderzand Spuihaven zuid"  # "Kornwerderzand Spuihaven zuid"; "Den Oever Stevinsluizen zuid"
# LOCATION_WIND = "Rotterdamse Hoek"
LOCATION_WIND = "Stavoren IJsselmeer"
# LOCATION_WATERHOOGTE = "Kornwerderzand, IJsselmeer binnenhaven"

# LOCATION_DEBIET_CARLIJN = "Stevin spuisluizen (afsluitdijk)" # "Lorentz spuisluizen (Afsluitdijk)"; "Stevin spuisluizen (afsluitdijk)"
# LOCATION_SALINITEIT = "Den Oever Stevinsluizen zuid" # "Kornwerderzand Spuihaven zuid"; "Den Oever Stevinsluizen zuid"
# LOCATION_WATERHOOGTE = "Den Oever, IJsselmeer binnenhaven"

YEARS = [2018, 2019, 2020, 2021, 2022]

# Define compass bins and labels (used for wind plots)
compass_directions = {
    "N": (337.5, 22.5),
    "NO": (22.5, 67.5),
    "O": (67.5, 112.5),
    "ZO": (112.5, 157.5),
    "Z": (157.5, 202.5),
    "ZW": (202.5, 247.5),
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


# %%
# Wind speed plot with wind direction coloring (0-360 continuous)

measurements_wind_daily_stavoren_with_msl = measurements_wind_daily.filter(
    (pl.col("Naam") == "Stavoren IJsselmeer")
    & (pl.col("Parameter_Wat_Omschrijving").str.contains("Mean Sea Level"))
).sort("time")

measurements_wind_daily_stavoren_without_msl = measurements_wind_daily.filter(
    (pl.col("Naam") == "Stavoren IJsselmeer")
    & ~(pl.col("Parameter_Wat_Omschrijving").str.contains("Mean Sea Level"))
).sort("time")

measurements_wind_daily_stavoren = (
    pl.concat(
        [
            measurements_wind_daily_stavoren_without_msl,
            measurements_wind_daily_stavoren_with_msl,
        ]
    )
    .unique(subset=["Naam", "time", "Grootheid.Code"], keep="first")
    .sort("time")
)

measurements_wind_daily_stavoren
# %%

# Get wind direction and speed data
wind_direction = (
    measurements_wind_daily_stavoren.filter(pl.col("Grootheid.Code") == "WINDRTG")
    .select(["Naam", "time", "Meetwaarde.Waarde_Numeriek"])
    .rename({"Meetwaarde.Waarde_Numeriek": "wind_direction"})
)

wind_speed = (
    measurements_wind_daily_stavoren.filter(pl.col("Grootheid.Code") == "WINDSHD")
    .select(["Naam", "time", "Meetwaarde.Waarde_Numeriek"])
    .rename({"Meetwaarde.Waarde_Numeriek": "wind_speed"})
)

# Join wind direction and speed data
wind_combined = wind_speed.join(wind_direction, on=["Naam", "time"], how="inner")

# Get unique station names
unique_stations = wind_combined.select("Naam").unique().sort("Naam")["Naam"].to_list()

# Create subplots for each station
fig, axes = plt.subplots(
    len(unique_stations), 1, figsize=(14, 4 * len(unique_stations))
)
if len(unique_stations) == 1:
    axes = [axes]

# Create color map for 8 compass directions with rainbow colors
colors = plt.cm.rainbow(np.linspace(0, 1, 8))
direction_colors = {
    direction: colors[i] for i, direction in enumerate(compass_directions.keys())
}

for idx, station in enumerate(unique_stations):
    station_data = (
        wind_combined.filter(pl.col("Naam") == station).sort("time").to_pandas()
    )

    # Bin wind directions
    station_data["wind_direction_bin"] = station_data["wind_direction"].apply(
        get_wind_direction_bin
    )

    # Map direction bins to numeric values for coloring
    direction_to_num = {
        direction: i for i, direction in enumerate(compass_directions.keys())
    }
    station_data["wind_direction_num"] = station_data["wind_direction_bin"].map(
        direction_to_num
    )

    ax = axes[idx]
    ax.plot(
        station_data["time"],
        station_data["wind_speed"],
        color="grey",
        linewidth=0.5,
        alpha=0.8,
        zorder=1,
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
        zorder=2,
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

# %%
# Interactive plot with wind direction arrows using Plotly with year dropdown

# Prepare data with year information
station_data["year"] = station_data["time"].dt.year
unique_years = sorted(station_data["year"].unique().tolist())

# Wind direction compass
compass_list = list(compass_directions.keys())
# Softer, eye-pleasing palette for the 8 compass directions (cyclic, muted tones)
_direction_palette = [
    "#5B8CE0",  # N  - blue
    "#6FB3D2",  # NO - teal-blue
    "#68D080",  # O  - green
    "#B0C65F",  # ZO - olive-green
    "#F0CE4E",  # Z  - warm yellow
    "#F09A5E",  # ZW - orange
    "#F06368",  # W  - red
    "#A190E0",  # NW - purple
]
color_map = {i: _direction_palette[i] for i in range(8)}

# Create figure with all years as traces (hidden except first)
fig = go.Figure()

for year_idx, year in enumerate(unique_years):
    year_data = station_data[station_data["year"] == year]

    if len(year_data) == 0:
        continue

    # Sort by time for better visualization
    year_data = year_data.sort_values("time")
    times = year_data["time"].to_list()
    wind_speeds = year_data["wind_speed"].to_numpy()
    wind_directions = year_data["wind_direction_num"].to_numpy()

    # Plot line
    fig.add_trace(
        go.Scatter(
            x=times,
            y=wind_speeds,
            mode="lines",
            name="Windsnelheid",
            line=dict(color="grey", width=1),
            opacity=0.5,
            hovertext=[compass_list[d] for d in wind_directions],
            hovertemplate="%{y:.2f} m/s %{hovertext}<extra></extra>",
            hoverlabel=dict(bgcolor=[color_map[d] for d in wind_directions]),
            visible=(year_idx == 0),
            legendgroup=str(year),
            showlegend=True,
        )
    )

    # Plot arrows showing wind direction
    arrow_angles = [180, 225, 270, 315, 0, 45, 90, 135]

    for direction_idx in range(8):
        mask = wind_directions == direction_idx
        x_points = np.array(times)[mask]
        y_points = wind_speeds[mask]
        direction_name = compass_list[direction_idx]

        fig.add_trace(
            go.Scatter(
                x=x_points,
                y=y_points,
                mode="markers",
                name=direction_name,
                marker=dict(
                    # size=11,
                    opacity=np.clip(y_points / 15, 0.8, 1.0),
                    symbol="arrow",
                    size=8 + y_points * 0.4,
                    color=color_map[direction_idx],
                    angle=arrow_angles[direction_idx],
                    # color=color_map[direction_idx],
                    line=dict(width=0.5 + y_points / 15, color="rgba(0,0,0,0.15)"),
                ),
                # hovertext=[
                #     times[i].strftime("%Y-%m-%d %H:%M") for i in np.where(mask)[0]
                # ],
                hoverinfo="skip",
                visible=(year_idx == 0),
                legendgroup=str(year),
                showlegend=True,
            )
        )

# Create dropdown buttons
buttons = []
for year_idx, year in enumerate(unique_years):
    # Calculate trace indices for this year (1 line + 8 direction markers = 9 traces per year)
    start_idx = year_idx * 9
    end_idx = start_idx + 9

    # Create visibility list
    visible = [False] * len(fig.data)
    for i in range(start_idx, end_idx):
        if i < len(fig.data):
            visible[i] = True

    buttons.append(
        dict(
            label=str(year),
            method="update",
            args=[
                {"visible": visible},
                {
                    "title.text": f"Wind snelheid met richting - {station} ({year})",
                },
            ],
        )
    )

fig.update_layout(
    updatemenus=[
        dict(
            type="dropdown",
            direction="down",
            x=0.04,
            y=1.08,
            showactive=True,
            buttons=buttons,
            bgcolor="white",
            borderwidth=0.5,
            pad={"r": 5, "t": 1},
            font={"size": 12},
        )
    ],
    title=f"Wind snelheid met richting - {station} ({unique_years[0]})",
    # xaxis_title="Time progression",
    yaxis_title="Wind snelheid [m/s]",
    hovermode="closest",
    template="plotly_white",
    height=600,
    width=1000,
    showlegend=True,
    legend=dict(title="Windrichting", x=1.0, y=1.0),
)

fig.update_yaxes(range=[0, station_data["wind_speed"].max() * 1.1])

output_dir = DATA_OUTPUT_DIR / "figuren" / "wind"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"wind_speed_direction_{station}.html"
fig.write_html(str(output_path))
fig.write_html(
    str(output_path),
    full_html=True,
    include_plotlyjs=True,
    default_height=600,
    default_width=1000,
)
print(f"Saved combined: {output_path}")
fig.show()

# %%
measurements_debiet_carlijn_daily

measurements_debiet_carlijn_daily_filt = measurements_debiet_carlijn_daily.filter(
    ~(pl.col("Naam").is_in(["Berging IJsselmeer"]))
).sort("time")
# %%

# Create daily interactive plot for inflows and outflows with netto flow
if measurements_debiet_carlijn_daily_filt is not None:
    # Define NAMEs that are leaving (outflows)
    outflows = [
        "Verdamping",
        "Stevin spuisluizen (afsluitdijk)",
        "Lorentz spuisluizen (Afsluitdijk)",
        "lek Spuisluizen (Afsluitdijk)",
        "PWN & WRK III",
        "inlaat Tacozijl (Friesland)",
        "inlaat Teroelsterkolk (Friesland)",
        "inlaat Hooglandgemaal (140)",
        "inlaat Workum",
        "Roggebotsluis",
        "inlaten Noordoostpolder (som)",
        "Krabbersgat spuisluizen (Houtribdijk) afvoer",
        "Houtrib spuisluizen (Houtribdijk) afvoer",
        "Zwarte Water afvoer",
        "Inlaat Vier Noorder Koggen (Westfriesland)",
    ]

    # Create a copy and categorize flows
    result_df_copy = measurements_debiet_carlijn_daily_filt.with_columns(
        [
            pl.col("time").dt.year().alias("Year"),
            pl.col("time").dt.date().alias("Datum"),
        ]
    )

    # Define IJssel inflow and Stevin/Lorentz outflows specifically
    ijssel_inflow = ["IJssel"]
    stevin_outflows = ["Stevin spuisluizen (afsluitdijk)"]
    lorentz_outflows = ["Lorentz spuisluizen (Afsluitdijk)"]

    def categorize_flow(name):
        if name in ijssel_inflow:
            return "IJssel"
        elif name in stevin_outflows:
            return "Stevinsluizen"
        elif name in lorentz_outflows:
            return "Lorentzsluizen"
        elif name in outflows:
            return "Overig uit"
        else:
            return "Overig in"

    result_df_copy = result_df_copy.with_columns(
        pl.col("Naam")
        .map_elements(categorize_flow, return_dtype=pl.Utf8)
        .alias("FlowType")
    )

    # Group by year, date and NAME, summing WAARDE
    daily_flows = result_df_copy.group_by(["Year", "Datum", "Naam", "FlowType"]).agg(
        pl.col("Meetwaarde.Waarde_Numeriek").sum()
    )

    # Get unique years for dropdown
    years = sorted(daily_flows["Year"].unique())

    # Define colors
    colors = {
        "IJssel": "darkblue",
        "Overig in": "lightblue",
        "Stevinsluizen": "darkred",
        "Lorentzsluizen": "red",
        "Overig uit": "lightsalmon",
    }

    # Create buttons for year dropdown
    buttons = []
    for year in years:
        year_data = daily_flows.filter(pl.col("Year") == year).sort("Datum")

        # Get date values for x-axis
        dates = sorted(year_data["Datum"].unique())

        # Create pivot tables for each category
        pivot_ijssel = (
            year_data.filter(pl.col("FlowType") == "IJssel")
            .group_by("Datum")
            .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
            .sort("Datum")
            .to_dict(as_series=False)
        )
        pivot_ijssel = {
            row[0]: row[1]
            for row in zip(
                pivot_ijssel["Datum"], pivot_ijssel["Meetwaarde.Waarde_Numeriek"]
            )
        }

        pivot_other_inflow = (
            year_data.filter(pl.col("FlowType") == "Overig in")
            .group_by("Datum")
            .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
            .sort("Datum")
            .to_dict(as_series=False)
        )
        pivot_other_inflow = {
            row[0]: row[1]
            for row in zip(
                pivot_other_inflow["Datum"],
                pivot_other_inflow["Meetwaarde.Waarde_Numeriek"],
            )
        }

        pivot_stevin = (
            year_data.filter(pl.col("FlowType") == "Stevinsluizen")
            .group_by("Datum")
            .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
            .sort("Datum")
            .to_dict(as_series=False)
        )
        pivot_stevin = {
            row[0]: -row[1]
            for row in zip(
                pivot_stevin["Datum"], pivot_stevin["Meetwaarde.Waarde_Numeriek"]
            )
        }

        pivot_lorentz = (
            year_data.filter(pl.col("FlowType") == "Lorentzsluizen")
            .group_by("Datum")
            .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
            .sort("Datum")
            .to_dict(as_series=False)
        )
        pivot_lorentz = {
            row[0]: -row[1]
            for row in zip(
                pivot_lorentz["Datum"], pivot_lorentz["Meetwaarde.Waarde_Numeriek"]
            )
        }

        pivot_other_outflow = (
            year_data.filter(pl.col("FlowType") == "Overig uit")
            .group_by("Datum")
            .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
            .sort("Datum")
            .to_dict(as_series=False)
        )
        pivot_other_outflow = {
            row[0]: -row[1]
            for row in zip(
                pivot_other_outflow["Datum"],
                pivot_other_outflow["Meetwaarde.Waarde_Numeriek"],
            )
        }

        # Calculate netto flow
        netto_flow = {
            d: pivot_ijssel.get(d, 0)
            + pivot_other_inflow.get(d, 0)
            + pivot_stevin.get(d, 0)
            + pivot_lorentz.get(d, 0)
            + pivot_other_outflow.get(d, 0)
            for d in dates
        }

        buttons.append(
            dict(
                label=str(year),
                method="update",
                args=[
                    {
                        "x": [dates] * 6,
                        "y": [
                            [pivot_ijssel.get(d, 0) for d in dates],
                            [pivot_other_inflow.get(d, 0) for d in dates],
                            [pivot_stevin.get(d, 0) for d in dates],
                            [pivot_lorentz.get(d, 0) for d in dates],
                            [pivot_other_outflow.get(d, 0) for d in dates],
                            [netto_flow.get(d, 0) for d in dates],
                        ],
                    },
                    {"title": f"Daily Inflows and Outflows - {year}"},
                ],
            )
        )

    # Create fresh figure with first year's data
    first_year = years[0]
    first_year_data = daily_flows.filter(pl.col("Year") == first_year).sort("Datum")

    dates = sorted(first_year_data["Datum"].unique())

    pivot_ijssel = (
        first_year_data.filter(pl.col("FlowType") == "IJssel")
        .group_by("Datum")
        .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
        .sort("Datum")
        .to_dict(as_series=False)
    )
    pivot_ijssel = {
        row[0]: row[1]
        for row in zip(
            pivot_ijssel["Datum"], pivot_ijssel["Meetwaarde.Waarde_Numeriek"]
        )
    }

    pivot_other_inflow = (
        first_year_data.filter(pl.col("FlowType") == "Overig in")
        .group_by("Datum")
        .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
        .sort("Datum")
        .to_dict(as_series=False)
    )
    pivot_other_inflow = {
        row[0]: row[1]
        for row in zip(
            pivot_other_inflow["Datum"],
            pivot_other_inflow["Meetwaarde.Waarde_Numeriek"],
        )
    }

    pivot_stevin = (
        first_year_data.filter(pl.col("FlowType") == "Stevinsluizen")
        .group_by("Datum")
        .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
        .sort("Datum")
        .to_dict(as_series=False)
    )
    pivot_stevin = {
        row[0]: -row[1]
        for row in zip(
            pivot_stevin["Datum"], pivot_stevin["Meetwaarde.Waarde_Numeriek"]
        )
    }

    pivot_lorentz = (
        first_year_data.filter(pl.col("FlowType") == "Lorentzsluizen")
        .group_by("Datum")
        .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
        .sort("Datum")
        .to_dict(as_series=False)
    )
    pivot_lorentz = {
        row[0]: -row[1]
        for row in zip(
            pivot_lorentz["Datum"], pivot_lorentz["Meetwaarde.Waarde_Numeriek"]
        )
    }

    pivot_other_outflow = (
        first_year_data.filter(pl.col("FlowType") == "Overig uit")
        .group_by("Datum")
        .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
        .sort("Datum")
        .to_dict(as_series=False)
    )
    pivot_other_outflow = {
        row[0]: -row[1]
        for row in zip(
            pivot_other_outflow["Datum"],
            pivot_other_outflow["Meetwaarde.Waarde_Numeriek"],
        )
    }

    netto_flow = {
        d: pivot_ijssel.get(d, 0)
        + pivot_other_inflow.get(d, 0)
        + pivot_stevin.get(d, 0)
        + pivot_lorentz.get(d, 0)
        + pivot_other_outflow.get(d, 0)
        for d in dates
    }

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_ijssel.get(d, 0) for d in dates],
            name="IJssel",
            marker=dict(color=colors["IJssel"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_other_inflow.get(d, 0) for d in dates],
            name="Overig in",
            marker=dict(color=colors["Overig in"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_stevin.get(d, 0) for d in dates],
            name="Stevinsluizen",
            marker=dict(color=colors["Stevinsluizen"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_lorentz.get(d, 0) for d in dates],
            name="Lorentzsluizen",
            marker=dict(color=colors["Lorentzsluizen"]),
        )
    )

    fig.add_trace(
        go.Bar(
            x=dates,
            y=[pivot_other_outflow.get(d, 0) for d in dates],
            name="Overig uit",
            marker=dict(color=colors["Overig uit"]),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[netto_flow.get(d, 0) for d in dates],
            name="Netto Flow",
            mode="lines",
            line=dict(color="black", width=1.5),
            yaxis="y",
        )
    )

    fig.update_layout(
        barmode="relative",
        title=f"Dagelijkse In- en Uitstromen - {first_year}",
        # xaxis_title="Date",
        yaxis_title="Debiet in miljoen m3 per dag",
        hovermode="x unified",
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.12,
                yanchor="top",
            )
        ],
        height=600,
        width=1200,
    )

    fig.write_html("interactive_flows.html")
    fig.show()

# %%
measurements_saliniteit_daily
# %%
# Combined subplot: Wind Speed Rotterdamse Hoek (top) + Wind Speed Stavoren (middle) + Daily Flows / Waterbalans (bottom)


if measurements_debiet_carlijn_daily_filt is not None:
    # --- Prepare per-station wind data ---
    # station_name_rh = "Rotterdamse Hoek"
    station_name_st = "Stavoren"

    def prepare_station_data(wind_combined_pl, station_name_filter):
        """Filter wind_combined for a station name substring and return a pandas df with year/direction columns.

        wind_combined_pl contains both WINDSNELHEID and WINDRICHTING rows, distinguished by Grootheid.Code.
        These are pivoted into wind_speed and wind_direction columns.
        """
        df_filtered = wind_combined_pl.filter(
            pl.col("Naam").str.contains(station_name_filter)
        )

        df_speed = df_filtered.filter(pl.col("Grootheid.Code") == "WINDSHD").select(
            ["time", "Naam", pl.col("Meetwaarde.Waarde_Numeriek").alias("wind_speed")]
        )
        df_dir = df_filtered.filter(pl.col("Grootheid.Code") == "WINDRTG").select(
            [
                "time",
                "Naam",
                pl.col("Meetwaarde.Waarde_Numeriek").alias("wind_direction"),
            ]
        )

        df_pivot = (
            df_speed.join(df_dir, on=["time", "Naam"], how="full", coalesce=True)
            .sort("time")
            .to_pandas()
        )

        df_pivot["wind_speed"] = df_pivot["wind_speed"].round(1)
        df_pivot["wind_direction"] = df_pivot["wind_direction"].round(1)
        df_pivot["wind_direction_bin"] = df_pivot["wind_direction"].apply(
            get_wind_direction_bin
        )
        direction_to_num = {d: i for i, d in enumerate(compass_directions.keys())}
        df_pivot["wind_direction_num"] = df_pivot["wind_direction_bin"].map(
            direction_to_num
        )
        df_pivot["year"] = df_pivot["time"].dt.year
        return df_pivot

    # station_data_rh = prepare_station_data(measurements_wind_daily, "Rotterdamse")
    # print(f"Rotterdamse Hoek data points: {len(station_data_rh)}")
    station_data_st = prepare_station_data(measurements_wind_daily, "Stavoren")
    print(f"Stavoren data points: {len(station_data_st)}")

    YEAR_MIN = 2018
    YEAR_MAX = 2022

    # unique_years_rh = sorted(station_data_rh["year"].unique().tolist()) if len(station_data_rh) > 0 else []
    unique_years_st = (
        sorted(
            y
            for y in station_data_st["year"].unique().tolist()
            if YEAR_MIN <= y <= YEAR_MAX
        )
        if len(station_data_st) > 0
        else []
    )

    # --- Prepare saliniteit data ---
    sal_df = measurements_saliniteit_daily.to_pandas()
    sal_df["time"] = sal_df["time"]
    sal_df["year"] = sal_df["time"].apply(
        lambda x: x.year if hasattr(x, "year") else None
    )
    sal_locations = sorted(sal_df["Naam"].unique().tolist())
    first_sal_location = sal_locations[0] if sal_locations else None

    fig_combined = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Daily Inflows and Outflows (Waterbalans)",
            "Wind Speed with Direction - Stavoren",
            "Saliniteit",
        ),
        vertical_spacing=0.01,
        row_heights=[0.4, 0.15, 0.25],
    )

    arrow_angles = [180, 225, 270, 315, 0, 45, 90, 135]

    traces_per_wind_year = 9  # 1 line + 8 directions
    # total_wind_rh_traces = len(unique_years_rh) * traces_per_wind_year

    # --- Wind traces Stavoren (row 2) ---
    for year_idx, year in enumerate(unique_years_st):
        year_data = station_data_st[station_data_st["year"] == year]
        if len(year_data) == 0:
            continue
        year_data = year_data.sort_values("time")
        times = year_data["time"].to_list()
        wind_speeds = year_data["wind_speed"].to_numpy()
        wind_directions = year_data["wind_direction_num"].to_numpy()

        fig_combined.add_trace(
            go.Scatter(
                x=times,
                y=wind_speeds,
                mode="lines",
                line=dict(color="#BFC8CD", width=0.8),
                opacity=0.4,
                # hoverinfo='skip',
                hovertext=[compass_list[d] for d in wind_directions],
                hovertemplate="%{y} m/s %{hovertext}<extra></extra>",
                hoverlabel=dict(bgcolor=[color_map[d] for d in wind_directions]),
                visible=(year_idx == 0),
                legendgroup=f"wind_st_{year}",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        for direction_idx in range(8):
            mask = wind_directions == direction_idx
            x_points = np.array(times)[mask]
            y_points = wind_speeds[mask]
            direction_name = compass_list[direction_idx]
            fig_combined.add_trace(
                go.Scatter(
                    x=x_points,
                    y=y_points,
                    mode="markers",
                    name=direction_name,
                    marker=dict(
                        # size=11,
                        opacity=np.clip(y_points / 15, 0.5, 1.0),
                        symbol="arrow",
                        size=8 + y_points * 0.4,
                        color=color_map[direction_idx],
                        angle=arrow_angles[direction_idx],
                        # color=color_map[direction_idx],
                        line=dict(width=0.5 + y_points / 15, color="rgba(0,0,0,0.15)"),
                    ),
                    # hovertext=[direction_name for _ in np.where(mask)[0]],
                    # hovertemplate='%{x|%m-%d} %{y} m/s %{hovertext}<extra></extra>',
                    hoverinfo="skip",
                    visible=(year_idx == 0),
                    legendgroup=f"wind_st_{year}",
                    legend="legend2",
                    showlegend=True,
                ),
                row=2,
                col=1,
            )

    fig_combined.update_layout(
        hovermode="closest",
        legend1=dict(title="Waterbalans"),
        legend2=dict(title="Stavoren"),
        legend3=dict(title="Saliniteit<br>op diepte"),
    )

    total_wind_st_traces = len(unique_years_st) * traces_per_wind_year
    total_wind_traces = total_wind_st_traces  # + total_wind_rh_traces

    # --- Flow / Waterbalans traces (row 1) ---
    flow_colors = {
        "Overig in": "#8CCFE3",
        "IJssel": "#3366A3",
        "Stevinsluizen": "#C44E52",
        "Lorentzsluizen": "#E98B39",
        "Overig uit": "#F5CDA7",
    }

    flow_years = sorted(
        y for y in daily_flows["Year"].unique() if YEAR_MIN <= y <= YEAR_MAX
    )

    for year_idx, year in enumerate(flow_years):
        year_data_flow = daily_flows.filter(pl.col("Year") == year).sort("Datum")
        dates_flow = sorted(year_data_flow["Datum"].unique())

        def get_pivot(df, flow_type, negate=False):
            d = (
                df.filter(pl.col("FlowType") == flow_type)
                .group_by("Datum")
                .agg(pl.col("Meetwaarde.Waarde_Numeriek").sum())
                .sort("Datum")
                .to_dict(as_series=False)
            )
            sign = -1 if negate else 1
            return {
                row[0]: sign * row[1]
                for row in zip(d["Datum"], d["Meetwaarde.Waarde_Numeriek"])
            }

        pv_ijssel = get_pivot(year_data_flow, "IJssel")
        pv_other_in = get_pivot(year_data_flow, "Overig in")
        pv_stevin = get_pivot(year_data_flow, "Stevinsluizen", negate=True)
        pv_lorentz = get_pivot(year_data_flow, "Lorentzsluizen", negate=True)
        pv_other_out = get_pivot(year_data_flow, "Overig uit", negate=True)
        netto = {
            d: pv_ijssel.get(d, 0)
            + pv_other_in.get(d, 0)
            + pv_stevin.get(d, 0)
            + pv_lorentz.get(d, 0)
            + pv_other_out.get(d, 0)
            for d in dates_flow
        }

        is_visible = year_idx == 0

        for legend_order, (flow_name, pivot, flow_color) in enumerate(
            [
                ("IJssel", pv_ijssel, flow_colors["IJssel"]),
                ("Overig in", pv_other_in, flow_colors["Overig in"]),
                ("Stevinsluizen", pv_stevin, flow_colors["Stevinsluizen"]),
                ("Lorentzsluizen", pv_lorentz, flow_colors["Lorentzsluizen"]),
                ("Overig uit", pv_other_out, flow_colors["Overig uit"]),
            ]
        ):
            fig_combined.add_trace(
                go.Bar(
                    x=dates_flow,
                    y=[pivot.get(d) if pivot.get(d) else None for d in dates_flow],
                    name=flow_name,
                    marker=dict(color=flow_color, line=dict(color="white", width=0.1)),
                    hovertemplate=f"%{{y:,.0f}} m³/d {flow_name}<extra></extra>",
                    visible=is_visible,
                    legendgroup=f"flow_{year}",
                    legend="legend",
                    legendrank=1000 + legend_order,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        fig_combined.add_trace(
            go.Scatter(
                x=dates_flow,
                y=[netto.get(d, 0) for d in dates_flow],
                name="Netto Flow",
                mode="lines",
                line=dict(color="#444444", width=1.3),
                hovertemplate="%{y:,.0f} m³/d Netto Flow<extra></extra>",
                visible=is_visible,
                legendgroup=f"flow_{year}",
                legend="legend",
                legendrank=1005,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    traces_per_flow_year = 6  # 5 bars + 1 netto line
    total_flow_traces = len(flow_years) * traces_per_flow_year

    # --- Saliniteit traces (row 4) ---
    # Red -> white -> blue colormap based on Bemonsteringshoogte (depth).
    all_depths = sorted(
        sal_df["WaarnemingMetadata.Bemonsteringshoogte"].dropna().unique().tolist()
    )

    depth_min = min(all_depths) if all_depths else 0.0
    depth_max = max(all_depths) if all_depths else 0.0

    def depth_to_color(depth, sorted_depths):
        """Map a depth value to a color based on its rank among the location's depths.

        Depths are ranked from most negative (deepest) to least negative.
        The most negative depth is always red, then progressively toward blue.
        - 4 depths: red, orange, lightblue, blue
        - 3 depths: red, white, blue
        - 2 depths: orange, blue
        - 1 depth: blue
        """
        # sorted_depths ascending (most negative first)
        n = len(sorted_depths)
        idx = sorted_depths.index(depth)
        if n >= 5:
            palette = ["#B2182B", "#EF8A62", "#9E9E9E", "#67A9CF", "#2166AC"]
        elif n == 4:
            palette = ["#B2182B", "#EF8A62", "#67A9CF", "#2166AC"]
        elif n == 3:
            palette = ["#B2182B", "#9E9E9E", "#2166AC"]
        elif n == 2:
            palette = ["#EF8A62", "#2166AC"]
        else:
            palette = ["#2166AC"]
        # For n > len(palette), clamp to last color
        return palette[min(idx, len(palette) - 1)]

    # Build saliniteit traces: one trace per location, all years combined
    # Traces are grouped by location; within each location there is one trace per year
    sal_years = sorted(
        y
        for y in sal_df["year"].dropna().astype(int).unique().tolist()
        if YEAR_MIN <= y <= YEAR_MAX
    )
    # (location, year) -> list of trace index offsets from sal_start
    sal_trace_index_map = {}

    sal_trace_offset = 0
    for loc_idx, location in enumerate(sal_locations):
        loc_df = sal_df[sal_df["Naam"] == location].sort_values("time")
        loc_years = sorted(loc_df["year"].dropna().astype(int).unique().tolist())
        for yr_idx, year in enumerate(loc_years):
            yr_df = loc_df[loc_df["year"] == year]
            depths = sorted(
                yr_df["WaarnemingMetadata.Bemonsteringshoogte"].unique().tolist()
            )
            for depth_idx, depth in enumerate(depths):
                # Color based on depth rank within this location's depths
                depth_color = depth_to_color(depth, depths)
                depth_df = yr_df[
                    yr_df["WaarnemingMetadata.Bemonsteringshoogte"] == depth
                ]
                # Break legend label into lines of at most 2 words each
                _name_words = f"{location} {depth}m".split()
                _sal_name = "<br>".join(
                    " ".join(_name_words[i : i + 2])
                    for i in range(0, len(_name_words), 2)
                )
                fig_combined.add_trace(
                    go.Scatter(
                        x=depth_df["time"].tolist(),
                        y=depth_df["Meetwaarde.Waarde_Numeriek"].tolist(),
                        mode="lines",
                        name=_sal_name,
                        line=dict(color=depth_color, width=1.5),
                        visible=False,  # all hidden initially; dropdown controls visibility
                        hovertemplate=f"%{{y:.0f}} mg/l {depth}m<extra></extra>",
                        legendgroup=f"sal_{location}_{year}_{depth}",
                        legend="legend3",
                        showlegend=True,
                    ),
                    row=3,
                    col=1,
                )
                sal_trace_index_map.setdefault((location, year), []).append(
                    sal_trace_offset
                )
                sal_trace_offset += 1

    total_sal_traces = sal_trace_offset
    total_traces = total_wind_traces + total_flow_traces + total_sal_traces

    # --- Independent dropdowns: year + saliniteit location ---
    # Waterbalans/wind respond to year only.
    # Saliniteit responds to both selected year and selected location.
    import json

    yaxis_title_font = dict(family="Arial", size=18)

    all_years = list(range(YEAR_MIN, YEAR_MAX + 1))

    sal_base = total_wind_traces + total_flow_traces

    # Absolute trace-index maps for JavaScript
    wind_indices_by_year = {}
    for year in unique_years_st:
        wind_st_year_idx = unique_years_st.index(year)
        wind_st_start = wind_st_year_idx * traces_per_wind_year
        wind_indices_by_year[int(year)] = list(
            range(wind_st_start, wind_st_start + traces_per_wind_year)
        )

    flow_indices_by_year = {}
    for year in flow_years:
        flow_year_idx = flow_years.index(year)
        flow_start = total_wind_traces + flow_year_idx * traces_per_flow_year
        flow_indices_by_year[int(year)] = list(
            range(flow_start, flow_start + traces_per_flow_year)
        )

    sal_indices_by_location_year = {}
    for (location, year), offsets in sal_trace_index_map.items():
        key = f"{location}||{int(year)}"
        sal_indices_by_location_year[key] = [sal_base + offset for offset in offsets]

    # Maximum salinity per location/year for dynamic y-axis scaling
    sal_max_by_location_year = {}
    for location in sal_locations:
        for year in sal_years:
            loc_year_sal = sal_df[
                (sal_df["Naam"] == location) & (sal_df["year"] == year)
            ]["Meetwaarde.Waarde_Numeriek"]

            if len(loc_year_sal) > 0:
                sal_max_by_location_year[f"{location}||{int(year)}"] = float(
                    loc_year_sal.max()
                )

    overall_sal_max = float(sal_df["Meetwaarde.Waarde_Numeriek"].max())

    def build_visible(year, sal_location):
        """Build visibility for the selected year and salinity location."""
        visible = [False] * total_traces

        # Wind: year only
        if year in unique_years_st:
            wind_st_year_idx = unique_years_st.index(year)
            wind_st_start = wind_st_year_idx * traces_per_wind_year

            for i in range(wind_st_start, wind_st_start + traces_per_wind_year):
                visible[i] = True

        # Waterbalans: year only
        if year in flow_years:
            flow_year_idx = flow_years.index(year)
            flow_start = total_wind_traces + flow_year_idx * traces_per_flow_year

            for i in range(flow_start, flow_start + traces_per_flow_year):
                visible[i] = True

        # Saliniteit: year + location
        if sal_location is not None:
            key = (sal_location, year)

            if key in sal_trace_index_map:
                for offset in sal_trace_index_map[key]:
                    visible[sal_base + offset] = True

        return visible

    # Defaults
    default_year = all_years[0] if all_years else None
    default_sal_location = sal_locations[0] if sal_locations else None

    # Initial visibility
    initial_visible = build_visible(default_year, default_sal_location)

    for trace_idx, vis in enumerate(initial_visible):
        fig_combined.data[trace_idx].visible = vis

    # Dropdown buttons emit events only.
    # The actual visibility logic is handled by JavaScript in the exported HTML.
    year_buttons = [
        dict(
            label=str(year),
            method="skip",
            args=[{"kind": "year", "value": int(year)}],
        )
        for year in all_years
    ]

    location_buttons = [
        dict(
            label=str(location),
            method="skip",
            args=[{"kind": "sal_location", "value": location}],
        )
        for location in sal_locations
    ]

    default_year_active = (
        all_years.index(default_year) if default_year in all_years else 0
    )

    default_location_active = (
        sal_locations.index(default_sal_location)
        if default_sal_location in sal_locations
        else 0
    )

    # Initial salinity axis range
    if default_sal_location is not None and default_year is not None:
        default_sal_key = f"{default_sal_location}||{int(default_year)}"
        max_sal = sal_max_by_location_year.get(default_sal_key, overall_sal_max)
    else:
        max_sal = overall_sal_max

    fig_combined.update_layout(
        height=1200,
        width=1000,
        title={
            "text": (
                f"IJsselmeer waterbalans, wind en saliniteit<br>"
                f"{default_year} - {default_sal_location}"
            ),
            "font": {"size": 18},
            "x": 0.53,
            "y": 0.94,
        },
        hovermode="x",
        template="plotly_white",
        barmode="relative",
        showlegend=True,
        paper_bgcolor="#F4F7F8",
        plot_bgcolor="#FAFCFC",
        legend=dict(x=1.02, y=1.0, yanchor="top"),
        legend2=dict(x=1.02, y=1.48 / 3, yanchor="top", font=dict(size=12)),
        legend3=dict(x=1.02, y=0.7 / 3, yanchor="top", font=dict(size=10)),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.1,
                y=1.04,
                showactive=True,
                active=default_year_active,
                buttons=year_buttons,
                bgcolor="white",
                bordercolor="black",
                borderwidth=0.5,
                pad={"r": 5, "t": 1},
                font={"size": 12},
            ),
            dict(
                type="dropdown",
                direction="down",
                x=0.43,
                y=0.285,
                showactive=True,
                active=default_location_active,
                buttons=location_buttons,
                bgcolor="white",
                bordercolor="black",
                borderwidth=0.5,
                pad={"r": 5, "t": 1},
                font={"size": 12},
            ),
        ],
        annotations=[
            dict(
                text="Jaar:",
                x=-0.03,
                y=1.04,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
            dict(
                text="Saliniteit locatie:",
                x=0.02,
                y=0.285,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left",
            ),
        ],
    )

    # Update y-axis titles for all subplots
    fig_combined.update_yaxes(
        title_text="Debiet [miljoen m³/dag]",
        title_font=yaxis_title_font,
        row=1,
        col=1,
    )

    fig_combined.update_yaxes(
        title_text="Windsnelheid [m/s]",
        title_font=yaxis_title_font,
        row=2,
        col=1,
    )

    # Initial salinity y-axis range
    fig_combined.update_yaxes(
        title_text="Concentratie [mg/l]",
        title_font=yaxis_title_font,
        row=3,
        col=1,
        range=[0, max_sal * 1.05],
    )

    # Initial x-axis range
    if default_year is not None:
        fig_combined.update_xaxes(
            range=[f"{default_year - 1}-12-24", f"{default_year + 1}-01-07"]
        )

    # Build a dictionary of empty combinations for JavaScript
    empty_combinations = {}
    for location in sal_locations:
        for year in sal_years:
            key = f"{location}||{int(year)}"
            year_int = int(year)
            if (
                key not in sal_indices_by_location_year
                or sal_indices_by_location_year[key] is None
                or len(sal_indices_by_location_year[key]) == 0
            ):
                if key not in empty_combinations:
                    empty_combinations[key] = True

    # JavaScript state manager for independent dropdowns
    post_script = f"""
    (function() {{
        const gd = document.getElementById('{{plot_id}}');

        const totalTraces = {total_traces};

        const windByYear = {json.dumps(wind_indices_by_year)};
        const flowByYear = {json.dumps(flow_indices_by_year)};
        const salByLocationYear = {json.dumps(sal_indices_by_location_year)};
        const salMaxByLocationYear = {json.dumps(sal_max_by_location_year)};
        const emptyLocationYearCombinations = {json.dumps(empty_combinations)};

        const overallSalMax = {overall_sal_max};
        const maxSal = {max_sal};

        const state = {{
            year: {json.dumps(int(default_year) if default_year is not None else None)},
            salLocation: {json.dumps(default_sal_location)}
        }};

        function addVisible(visible, indices) {{
            if (!indices) return;

            for (const idx of indices) {{
                visible[idx] = true;
            }}
        }}

        function updateFigure() {{
            const visible = Array(totalTraces).fill(false);

            const yearKey = String(state.year);
            const salKey = state.salLocation + "||" + String(state.year);

            // Jaar stuurt waterbalans en wind
            addVisible(visible, windByYear[yearKey]);
            addVisible(visible, flowByYear[yearKey]);

            // Jaar + locatie sturen saliniteit
            addVisible(visible, salByLocationYear[salKey]);

            const salMax = salMaxByLocationYear[salKey] || overallSalMax;

            // Check if current combination is empty
            const hasData = salByLocationYear[salKey] && salByLocationYear[salKey].length > 0;

            // Build annotations - only show "Geen data beschikbaar" if no data
            const annotations = gd.layout.annotations.filter(ann => ann.text !== "Geen data beschikbaar");
            if (!hasData) {{
                annotations.push({{
                    text: "Geen data beschikbaar",
                    xref: "paper",
                    yref: "paper",
                    x: 0.5,
                    y: (gd.layout.yaxis3.domain[0] + gd.layout.yaxis3.domain[1]) / 2,
                    showarrow: false,
                    font: {{size: 48, color: "rgba(200, 200, 200, 0.3)"}},
                    xanchor: "center",
                    yanchor: "middle"
                }});
            }}

            const layoutUpdate = {{
                "title.text": "IJsselmeer waterbalans, wind en saliniteit<br>" +
                            state.year + " - " + state.salLocation,
                "xaxis.range": [
                    String(state.year - 1) + "-12-24",
                    String(state.year + 1) + "-01-07"
                ],
                "xaxis2.range": [
                    String(state.year - 1) + "-12-24",
                    String(state.year + 1) + "-01-07"
                ],
                "xaxis3.range": [
                    String(state.year - 1) + "-12-24",
                    String(state.year + 1) + "-01-07"
                ],
                "yaxis3.range": [0, salMax * 1.05],
                "annotations": annotations
            }};

            Plotly.update(gd, {{visible: visible}}, layoutUpdate);
        }}

        gd.on("plotly_buttonclicked", function(eventData) {{
            if (!eventData || !eventData.button || !eventData.button.args) return;

            const payload = eventData.button.args[0];

            if (payload.kind === "year") {{
                state.year = payload.value;
            }}

            if (payload.kind === "sal_location") {{
                state.salLocation = payload.value;
            }}

            updateFigure();
        }});

        updateFigure();
    }})();
    """

    combined_output_path = (
        output_dir / f"interactive_plot_waterbalans_wind_saliniteit_{TODAY}.html"
    )

    fig_combined.write_html(
        str(combined_output_path),
        full_html=True,
        include_plotlyjs=True,
        post_script=post_script,
        default_height=1200,
        default_width=1000,
    )

    print(f"Saved combined: {combined_output_path}")
    fig_combined.show()

# %%

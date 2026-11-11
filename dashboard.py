import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sea Level Rise Dashboard", layout="wide")

GLOBAL_FILE = "global_mean_sea_level_1993-2024.csv"
OCEAN_FILES = {
    "Atlantic": "atlantic.csv",
    "Pacific": "pacific.csv",
    "Indian": "indian.csv",
}
OCEAN_COLORS = {
    "Atlantic": "#89CFF0",
    "Pacific": "#F4E1C1",
    "Indian": "#C4EBA8",
}


@st.cache_data
def load_global():
    data = pd.read_csv(GLOBAL_FILE)
    data["Year"] = data["YearPlusFraction"].astype(int)
    data["Month"] = ((data["YearPlusFraction"] % 1) * 12 + 1).astype(int)
    data["RollingAvg"] = data["SmoothedGMSLWithGIA"].rolling(window=12).mean()
    return data


@st.cache_data
def load_ocean(file, name):
    data = pd.read_csv(file, comment="#")
    satellite_cols = data.columns[1:]
    data[satellite_cols] = data[satellite_cols].apply(pd.to_numeric, errors="coerce")
    data["SeaLevel"] = data[satellite_cols].mean(axis=1)
    data = data[["year", "SeaLevel"]].copy()
    data.columns = ["Year", "SeaLevel"]
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data = data.dropna()
    data["Ocean"] = name
    return data


df = load_global()
ocean_df = pd.concat([load_ocean(f, n) for n, f in OCEAN_FILES.items()])

st.title("Sea Level Rise Dashboard")
st.markdown('<div id="top"></div>', unsafe_allow_html=True)

st.markdown("""
This interactive dashboard visualises **global and regional sea level trends from 1993–2024** 
using satellite observations. Users can explore long-term trends, yearly changes, 
statistical distributions, and regional ocean comparisons to better understand the 
impact of climate change on sea levels.
""")

st.info("""
Sea level rise is a major indicator of **global climate change**. 
Rising ocean temperatures cause **thermal expansion**, while melting glaciers 
and ice sheets contribute additional water to the oceans. Even small increases 
in sea levels can significantly impact coastal communities, ecosystems, 
and global weather patterns.
""")

st.sidebar.header("Dashboard Controls")

st.sidebar.markdown("""
Use these controls to explore different **oceans and time periods**.
""")

dataset = st.sidebar.selectbox(
    "Select Ocean",
    ["Global", "Atlantic", "Pacific", "Indian"],
    format_func=lambda x: {
        "Global": "Global",
        "Atlantic": "Atlantic Ocean",
        "Pacific": "Pacific Ocean",
        "Indian": "Indian Ocean",
    }[x],
)

st.sidebar.markdown("""
You can analyse:

• **Global sea level rise**  
• **Atlantic Ocean trends**  
• **Pacific Ocean trends**  
• **Indian Ocean trends**
""")

st.sidebar.markdown("""
Adjust the year range to observe how sea levels change over time.
""")

year_range = st.sidebar.slider("Select Year Range", 1993, 2024, (1993, 2024))
start, end = year_range

# Filter on "< end + 1" so the final selected year is included in full
# (the fractional years of 2024 are greater than 2024.0).
if dataset == "Global":
    value_col = "SmoothedGMSLWithGIA"
    x_col = "YearPlusFraction"
    name = "Global"
    color = "darkcyan"
    filtered = df[(df[x_col] >= start) & (df[x_col] < end + 1)].copy()
else:
    value_col = "SeaLevel"
    x_col = "Year"
    name = f"{dataset} Ocean"
    color = OCEAN_COLORS[dataset]
    filtered = ocean_df[
        (ocean_df["Ocean"] == dataset)
        & (ocean_df["Year"] >= start)
        & (ocean_df["Year"] < end + 1)
    ].copy()

if len(filtered) < 2:
    st.warning("Not enough data in this range. Please widen the year range.")
    st.stop()

filtered["YearInt"] = filtered[x_col].astype(int)

# Trend and cumulative rise are calculated on the selected range,
# so every number on the page matches the slider.
slope, intercept = np.polyfit(filtered[x_col], filtered[value_col], 1)
filtered["Trend"] = slope * filtered[x_col] + intercept
filtered["CumulativeRise"] = filtered[value_col] - filtered[value_col].iloc[0]

# True year-by-year figures: average within each calendar year, then take
# the difference between consecutive years.
yearly = filtered.groupby("YearInt", as_index=False)[value_col].mean()
yearly["YearlyChange"] = yearly[value_col].diff()
yearly["RollingAvg5"] = yearly[value_col].rolling(window=5).mean()

total = filtered["CumulativeRise"].iloc[-1]
latest = filtered[value_col].iloc[-1]

# ---------- Charts ----------
if dataset == "Global":
    fig1 = px.line(
        filtered,
        x=x_col,
        y=value_col,
        title="Global Sea Level Rise",
        labels={x_col: "Year", value_col: "Sea Level (mm)"},
        color_discrete_sequence=[color],
        hover_data={"Year": True, "Month": True, value_col: ":.2f"},
    )
    fig1.add_scatter(
        x=filtered[x_col],
        y=filtered["RollingAvg"],
        mode="lines",
        line=dict(color="orange"),
        name="Moving Average",
        hovertemplate="Year: %{x:.1f}<br>Moving Avg: %{y:.2f} mm",
    )
else:
    fig1 = px.line(
        filtered,
        x=x_col,
        y=value_col,
        title=f"{name} Level Rise",
        labels={x_col: "Year", value_col: "Ocean Level (mm)"},
        color_discrete_sequence=[color],
        hover_data={"Year": True, value_col: ":.2f"},
    )

fig1.add_scatter(
    x=filtered[x_col],
    y=filtered["Trend"],
    mode="lines",
    line=dict(color="red", dash="dash"),
    name="Trend",
    hovertemplate="Year: %{x:.1f}<br>Trend: %{y:.2f} mm",
)

fig2 = px.bar(
    yearly.dropna(subset=["YearlyChange"]),
    x="YearInt",
    y="YearlyChange",
    title=f"{name} Yearly Change in Sea Level",
    labels={"YearInt": "Year", "YearlyChange": "Change (mm)"},
    color_discrete_sequence=[color],
    hover_data={"YearlyChange": ":.2f"},
)

fig3 = px.histogram(
    filtered,
    x=value_col,
    nbins=30,
    title=f"{name} Distribution of Sea Level Values",
    labels={value_col: "Sea Level (mm)"},
    color_discrete_sequence=[color],
    opacity=0.75,
)
fig3.update_traces(
    marker_line_color="black",
    marker_line_width=1,
    hovertemplate="Sea Level: %{x:.2f} mm<br>Frequency: %{y}",
)
fig3.update_layout(xaxis_title="Sea Level (mm)", yaxis_title="Frequency", bargap=0)
fig3.add_vline(x=filtered[value_col].mean(), line_dash="dash", line_color="red")
fig3.add_scatter(
    x=[None],
    y=[None],
    mode="lines",
    line=dict(color="red", dash="dash"),
    name="Mean",
)

if dataset == "Global":
    heatmap_data = filtered.pivot_table(values=value_col, index="Month", columns="Year")
    fig4 = px.imshow(
        heatmap_data,
        title="Sea Level Heatmap (Month vs Year)",
        color_continuous_scale="Teal",
        labels=dict(x="Year", y="Month", color="Sea Level (mm)"),
    )
else:
    fig4 = px.line(
        yearly,
        x="YearInt",
        y="RollingAvg5",
        title=f"{name} Smoothed Trend (5-Year Avg)",
        labels={"YearInt": "Year", "RollingAvg5": "Ocean Level (mm)"},
        color_discrete_sequence=[color],
        hover_data={"RollingAvg5": ":.2f"},
    )

fig5 = px.area(
    filtered,
    x=x_col,
    y="CumulativeRise",
    title=f"{name} Total Sea Level Rise",
    labels={x_col: "Year", "CumulativeRise": "Total Rise (mm)"},
    color_discrete_sequence=[color],
    hover_data={"Year": True, "CumulativeRise": ":.2f"},
)

# ---------- Summary metrics ----------
col1, col2, col3 = st.columns(3)
col1.metric("Total Sea Level Rise", f"{total:.2f} mm")
col2.metric("Average Annual Change", f"{slope:.2f} mm/year")
col3.metric("Most Recent Sea Level", f"{latest:.2f} mm")

direction = "an upward" if slope > 0 else "a downward"
st.success(
    f"Between **{start} and {end}**, a linear fit to the data gives an average change "
    f"of approximately **{slope:.2f} mm per year**, indicating {direction} trend "
    "over the selected period."
)

st.plotly_chart(fig1, width="stretch")
st.plotly_chart(fig2, width="stretch")
st.plotly_chart(fig3, width="stretch")
st.plotly_chart(fig4, width="stretch")
st.plotly_chart(fig5, width="stretch")

st.markdown("---")
st.subheader("Key Insights")

st.markdown("""
• Global sea levels show an **upward trend over 1993–2024**.  
• Yearly changes vary from year to year, around a **rising long-term trend**.  
• Regional ocean trends show **similar rising patterns across the Atlantic, Pacific, and Indian Oceans**.  
• The cumulative rise highlights the **long-term impact of climate change on global oceans**.
""")

st.markdown("---")
st.subheader("Dataset Overview")

colA, colB, colC = st.columns(3)
colA.metric("Records", len(filtered))
colB.metric("Start Year", int(filtered["YearInt"].min()))
colC.metric("End Year", int(filtered["YearInt"].max()))

st.markdown("---")
st.subheader("Filtered Dataset Preview")
st.dataframe(filtered)

st.markdown("---")
st.subheader("Data Sources")

st.markdown("""
This dashboard uses publicly available sea level datasets collected from satellite observations.

Global Mean Sea Level Dataset (1993–2024)  
Source: NASA / NOAA satellite altimetry data  
https://www.kaggle.com/datasets/yunichie/global-mean-sea-level-1993-2024

Regional Ocean Sea Level Data (Atlantic, Pacific, Indian)  
Source: NOAA Laboratory for Satellite Altimetry  
https://www.star.nesdis.noaa.gov/socd/lsa/SeaLevelRise/LSA_SLR_timeseries.php
""")

st.markdown("---")

st.caption("""
Dashboard created for COMP3089 Data Visualisation Project  
Theme: Climate Change and Its Impact Globally
""")

st.markdown("""
<a href="#top" style="
position: fixed;
bottom: 40px;
right: 40px;
background-color: darkcyan;
color: white;
padding: 10px 15px;
border-radius: 10px;
text-decoration: none;
font-weight: bold;
">
⬆ Back to Top
</a>
""", unsafe_allow_html=True)

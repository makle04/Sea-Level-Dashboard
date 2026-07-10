import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Sea Level Rise Dashboard", layout="wide")

df = pd.read_csv("global_mean_sea_level_1993-2024.csv")

df["YearlyChange"] = df["SmoothedGMSLWithGIA"].diff()
df["Acceleration"] = df["YearlyChange"].diff()
df["CumulativeRise"] = df["SmoothedGMSLWithGIA"] - df["SmoothedGMSLWithGIA"].iloc[0]

df["Year"] = df["YearPlusFraction"].astype(int)
df["Month"] = ((df["YearPlusFraction"] % 1) * 12 + 1).astype(int)

coeffs = np.polyfit(df["YearPlusFraction"], df["SmoothedGMSLWithGIA"], 1)
trend = np.poly1d(coeffs)
df["Trend"] = trend(df["YearPlusFraction"])
df["RollingAvg"] = df["SmoothedGMSLWithGIA"].rolling(window=12).mean()

def load_ocean(file, name):
    df = pd.read_csv(file, comment="#")
    satellite_cols = df.columns[1:]
    df[satellite_cols] = df[satellite_cols].apply(pd.to_numeric, errors="coerce")
    df["SeaLevel"] = df[satellite_cols].mean(axis=1)
    df = df[["year", "SeaLevel"]].copy()
    df.columns = ["Year", "SeaLevel"]
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.dropna()
    df["Ocean"] = name
    return df

atlantic = load_ocean("atlantic.csv", "Atlantic")
pacific = load_ocean("pacific.csv", "Pacific")
indian = load_ocean("indian.csv", "Indian")

ocean_df = pd.concat([atlantic, pacific, indian])

ocean_colors = {
    "Atlantic": "#89CFF0",
    "Pacific": "#F4E1C1",
    "Indian": "#C4EBA8"
}

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
        "Indian": "Indian Ocean"
    }[x]
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

year_range = st.sidebar.slider(
    "Select Year Range",
    1993,
    2024,
    (1993, 2024)
)

if dataset == "Global":

    filtered = df[
        (df["YearPlusFraction"] >= year_range[0]) &
        (df["YearPlusFraction"] <= year_range[1])
    ]

    fig1 = px.line(
        filtered,
        x="YearPlusFraction",
        y="SmoothedGMSLWithGIA",
        title="Global Sea Level Rise",
        labels={
            "YearPlusFraction": "Year",
            "SmoothedGMSLWithGIA": "Sea Level (mm)"
        },
        color_discrete_sequence=["darkcyan"],
        hover_data={
            "Year":True,
            "Month":True,
            "SmoothedGMSLWithGIA":":.2f"
        }
    )

    fig1.add_scatter(
        x=filtered["YearPlusFraction"],
        y=filtered["Trend"],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Trend",
        hovertemplate="Year: %{x:.0f}<br>Month: %{customdata[1]}<br>Trend: %{y:.2f} mm",
        customdata=filtered[["Year","Month"]]
    )

    fig1.add_scatter(
        x=filtered["YearPlusFraction"],
        y=filtered["RollingAvg"],
        mode="lines",
        line=dict(color="orange"),
        name="Moving Average",
        hovertemplate="Year: %{x:.0f}<br>Month: %{customdata[1]}<br>Moving Avg: %{y:.2f} mm",
        customdata=filtered[["Year","Month"]]
    )

    fig2 = px.bar(
        filtered,
        x="YearPlusFraction",
        y="YearlyChange",
        title="Yearly Change in Sea Level",
        labels={
            "YearPlusFraction": "Year",
            "YearlyChange": "Change (mm)"
        },
        color_discrete_sequence=["darkcyan"],
        hover_data={
            "Year":True,
            "Month":True,
            "YearlyChange":":.2f"
        }
    )

    fig3 = px.histogram(
        filtered,
        x="SmoothedGMSLWithGIA",
        nbins=30,
        title="Distribution of Sea Level Values",
        labels={"SmoothedGMSLWithGIA": "Sea Level (mm)"},
        color_discrete_sequence=["darkcyan"],
        opacity=0.75
    )

    fig3.update_traces(
        marker_line_color="black",
        marker_line_width=1,
        hovertemplate="Sea Level: %{x:.2f} mm<br>Frequency: %{y}"
    )

    fig3.update_layout(
        xaxis_title="Sea Level (mm)",
        yaxis_title="Frequency",
        bargap=0
    )

    mean_val = filtered["SmoothedGMSLWithGIA"].mean()

    fig3.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red"
    )

    fig3.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Mean"
    )

    heatmap_data = filtered.pivot_table(
        values="SmoothedGMSLWithGIA",
        index="Month",
        columns="Year"
    )

    fig4 = px.imshow(
        heatmap_data,
        title="Sea Level Heatmap (Month vs Year)",
        color_continuous_scale="Teal",
        labels=dict(x="Year", y="Month", color="Sea Level (mm)")
    )

    fig5 = px.area(
        filtered,
        x="YearPlusFraction",
        y="CumulativeRise",
        title="Total Sea Level Rise",
        labels={
            "YearPlusFraction": "Year",
            "CumulativeRise": "Total Rise (mm)"
        },
        color_discrete_sequence=["darkcyan"],
        hover_data={
            "Year":True,
            "Month":True,
            "CumulativeRise":":.2f"
        }
    )

    total = filtered["CumulativeRise"].iloc[-1]
    avg = filtered["YearlyChange"].mean()
    latest = filtered["SmoothedGMSLWithGIA"].iloc[-1]

else:

    filtered = ocean_df[
        (ocean_df["Ocean"] == dataset) &
        (ocean_df["Year"] >= year_range[0]) &
        (ocean_df["Year"] <= year_range[1])
    ].copy()
    filtered["Month"] = 1
    color = ocean_colors[dataset]

    filtered["YearlyChange"] = filtered["SeaLevel"].diff()
    filtered["CumulativeRise"] = filtered["SeaLevel"] - filtered["SeaLevel"].iloc[0]

    coeffs = np.polyfit(filtered["Year"], filtered["SeaLevel"], 1)
    trend = np.poly1d(coeffs)

    filtered["Trend"] = trend(filtered["Year"])
    filtered["RollingAvg"] = filtered["SeaLevel"].rolling(window=5).mean()

    fig1 = px.line(
        filtered,
        x="Year",
        y="SeaLevel",
        title=f"{dataset} Ocean Level Rise",
        labels={
            "Year": "Year",
            "SeaLevel": "Ocean Level (mm)"
        },
        color_discrete_sequence=[color],
        hover_data={
            "Year":True,
            "SeaLevel":":.2f"
        }
    )

    fig1.add_scatter(
        x=filtered["Year"],
        y=filtered["Trend"],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Trend",
        hovertemplate="Year: %{x}<br>Trend Level: %{y:.2f} mm"
    )

    fig2 = px.bar(
        filtered,
        x="Year",
        y="YearlyChange",
        title=f"{dataset} Ocean Yearly Change",
        labels={
            "Year": "Year",
            "YearlyChange": "Change (mm)"
        },
        color_discrete_sequence=[color],
        hover_data={"Year":True,"YearlyChange":":.2f"}
    )

    fig3 = px.histogram(
        filtered,
        x="SeaLevel",
        nbins=30,
        title=f"{dataset} Ocean Distribution",
        labels={"SeaLevel": "Ocean Level (mm)"},
        color_discrete_sequence=[color],
        opacity=0.75
    )

    fig3.update_traces(
        marker_line_color="black",
        marker_line_width=1,
        hovertemplate="Ocean Level: %{x:.2f} mm<br>Frequency: %{y}"
    )

    fig3.update_layout(
        xaxis_title="Sea Level (mm)",
        yaxis_title="Frequency",
        bargap=0
    )

    mean_val = filtered["SeaLevel"].mean()

    fig3.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red"
    )

    fig3.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        line=dict(color="red", dash="dash"),
        name="Mean"
    )

    fig4 = px.line(
        filtered,
        x="Year",
        y="RollingAvg",
        title=f"{dataset} Ocean Smoothed Trend (5-Year Avg)",
        labels={
            "Year": "Year",
            "RollingAvg": "Ocean Level (mm)"
        },
        color_discrete_sequence=[color],
        hover_data={"Year":True,"RollingAvg":":.2f"}
    )

    fig5 = px.area(
        filtered,
        x="Year",
        y="CumulativeRise",
        title=f"{dataset} Ocean Total Rise",
        labels={
            "Year": "Year",
            "CumulativeRise": "Total Rise (mm)"
        },
        color_discrete_sequence=[color],
        hover_data={"Year":True,"CumulativeRise":":.2f"}
    )

    total = filtered["CumulativeRise"].iloc[-1]
    avg = filtered["YearlyChange"].mean()
    latest = filtered["SeaLevel"].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Total Sea Level Rise", f"{total:.2f} mm")
col2.metric("Average Annual Change", f"{avg:.2f} mm")
col3.metric("Most Recent Sea Level", f"{latest:.2f} mm")

trend_rate = filtered["YearlyChange"].mean()

st.success(
f"Between **{year_range[0]} and {year_range[1]}**, the average sea level rise "
f"is approximately **{trend_rate:.2f} mm per year**, indicating a consistent "
"long-term upward trend in global ocean levels."
)

st.plotly_chart(fig1, width="stretch")
st.plotly_chart(fig2, width="stretch")
st.plotly_chart(fig3, width="stretch")
st.plotly_chart(fig4, width="stretch")
st.plotly_chart(fig5, width="stretch")

st.markdown("---")
st.subheader("Key Insights")

st.markdown("""
• Global sea levels have shown a **consistent upward trend since 1993**.  
• Yearly changes indicate **variability but overall acceleration**.  
• Regional ocean trends show **similar rising patterns across Atlantic, Pacific, and Indian Oceans**.  
• The cumulative rise highlights the **long-term impact of climate change on global oceans**.
""")

st.markdown("---")
st.subheader("Dataset Overview")

colA, colB, colC = st.columns(3)

colA.metric("Records", len(filtered))
colB.metric("Start Year", int(filtered["Year"].min()))
colC.metric("End Year", int(filtered["Year"].max()+1))

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
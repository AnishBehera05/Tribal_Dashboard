import plotly.express as px
import pandas as pd
import json

# Load GeoJSON and DataFrame
with open("NFHS5_statefiles.geojson", "r") as f:
    geojson = json.load(f)

df = pd.read_csv("State_Level_Tribal_Health_Factsheet_India_NFHS_V.csv", encoding="ISO-8859-1")

# Filter data for one indicator
indicator = "Households having access to internet (%)"
df = df[df["indicator_name"] == indicator]
df["Total"] = pd.to_numeric(df["Total"], errors="coerce")

# Check if state_acronym exists in GeoJSON
print("GeoJSON example:", geojson["features"][0]["properties"])  # ensure 'state_acronym' exists

# Plot map
fig = px.choropleth(
    df,
    geojson=geojson,
    locations="state_acronym",
    color="Total",
    featureidkey="properties.state_acronym",
    color_continuous_scale="Viridis",
    title=indicator
)

fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
fig.show()

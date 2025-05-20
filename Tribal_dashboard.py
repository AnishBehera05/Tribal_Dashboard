import dash
from dash import ctx
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, ALL, State
import pandas as pd
import plotly.express as px
import json
from fpdf import FPDF
import tempfile
import plotly.graph_objects as go
import ollama
import re

state_csv = "State_Level_Tribal_Health_Factsheet_India_NFHS_V.csv"
district_csv = "District_Level_Tribal_Health_Factsheet_India_NFHS_V.csv"
df_state = pd.read_csv(state_csv, encoding='ISO-8859-1')
df_district = pd.read_csv(district_csv, encoding='ISO-8859-1')

with open("NFHS5_statefiles.geojson", "r") as f:
    geojson_states = json.load(f)
with open("NFHS5_districtlevel.geojson", "r") as f:
    geojson_districts = json.load(f)

df_district["district_id"] = df_district["district_id"].astype(str).str.strip().str.title()
for feature in geojson_districts["features"]:
    feature["properties"]["district_id"] = str(feature["properties"]["district_id"]).strip().title()

state_data = {state: df_state[df_state["state_acronym"] == state] for state in df_state["state_acronym"].unique()}
state_name_mapping = df_state[['state_acronym', 'state_name']].drop_duplicates().set_index('state_acronym')['state_name'].to_dict()
indicator_meta = pd.read_csv("indicators.csv", encoding='ISO-8859-1')
indicator_type_map = dict(zip(indicator_meta['indicator_name'], indicator_meta['categories']))

df_state["state_name"] = df_state["state_name"].replace(
    "Dadra and Nagar Haveli and Daman and Diu", "D&N Haveli & Daman & Diu"
)
df_district["state_name"] = df_district["state_name"].replace(
    "Dadra and Nagar Haveli and Daman and Diu", "D&N Haveli & Daman & Diu"
)

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.enable_dev_tools(debug=False)

config_no_buttons = {"displayModeBar": False}
    
app.layout = html.Div([

    dcc.Store(id='main-tabs', data='map-tab'),

    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col(
                    html.H2("Tribal Health FactSheet Dashboard", className="text-white fw-bold mb-0"),
                    md="auto"
                ),
                dbc.Col(
                    dbc.Nav([
                        dbc.NavLink("🗺️ Map View", href="#", active="exact", id="tab-map-tab"),
                        dbc.NavLink("📊 Bar Graph", href="#", active="exact", id="tab-bar-tab"),
                        dbc.NavLink("🎻 Violin Plot", href="#", active="exact", id="tab-violin-tab"),
                        dbc.NavLink("🫧 Bubble Plot", href="#", active="exact", id="tab-bubble-tab"),
                    ], pills=False, justified=False, navbar=True, className="ms-auto text-white"),
                    width="auto"
                )
            ], align="center", justify="between")
        ], fluid=True)
    ], style={
        'backgroundColor': '#2C3E50',
        'position': 'fixed',
        'top': '0',
        'zIndex': '1000',
        'width': '100%',
        'boxShadow': '0 2px 6px rgba(0,0,0,0.3)',
        'padding': '15px 20px'
    }),

    html.Div(style={'height': '100px'}),

    dbc.Container([

        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='state-selection',
                    options=[{'label': state_name_mapping[state], 'value': state} for state in state_data.keys()],
                    value=None,
                    placeholder="Select a state",
                    clearable=True
                ),
                width=4
            ),
            dbc.Col(
                dcc.RadioItems(
                    id='category-selection-type',
                    options=[
                        {'label': 'ST', 'value': 'ST'},
                        {'label': 'Non-ST', 'value': 'Non-ST'},
                        {'label': 'Total', 'value': 'Total'}
                    ],
                    value='Total',
                    inline=True,
                    labelStyle={'marginRight': '10px'}
                ),
                width=4,
                style={"textAlign": "center"}
            ),
            dbc.Col(
                html.Div([
                    dbc.Button("Reset", id="reset", color="primary")
                ], className="d-flex justify-content-end"),
                width=4
            )
        ], justify="between", className="mb-3", style={"padding": "10px 0"}),

        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='indicator-category-dropdown',
                    options=[{'label': cat, 'value': cat} for cat in sorted(df_state['category'].dropna().unique())],
                    value='Access to Communication/Mass Media',
                    placeholder="Select Category"
                )
            ], width=6),
            dbc.Col(
                html.Div([
                    dbc.Button("Download", id="download-btn", color="success")
                ], className="d-flex justify-content-end"),
                width=6
            )
        ], justify="between", className="mb-3"),

        html.Div(id='indicator-selectors'),

        html.H3(id='selected-state-name', style={'display': 'none'}),

        dcc.Download(id="download-figures"),
        dcc.Store(id='clicked-state-store'),
        html.Div(id='visualization-panel'),

    ], fluid=True),

    html.Div([
        dbc.Container(
            html.P("© 2025 PopulationCouncil Consulting",
                   className="text-white text-center mb-0",
                   style={"padding": "10px", "fontSize": "14px"})
        )
    ], style={
        "backgroundColor": "#2C3E50",
        "marginTop": "40px"
    })

], style={
    'margin': '0',
    'padding': '0',
    'width': '100%',
    'overflowX': 'hidden'
})

@app.callback(
    Output("download-figures", "data"),
    Input("download-btn", "n_clicks"),
    State('main-tabs', 'data'),
    State('category-selection-type', 'value'),
    State('indicator-category-dropdown', 'value'),
    State({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
    State('state-selection', 'value'),
    prevent_initial_call=True
)
def download_all_figures(n_clicks, active_tab, hh_type, category, indicators, selected_state):
    indicators = indicators + [None] * (4 - len(indicators))

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    width, height = 297, 210

    with tempfile.TemporaryDirectory() as tmpdir:
        if active_tab == "map-tab":
            figs = update_all_maps(hh_type, category, indicators, selected_state)
            for i, fig in enumerate(figs):
                if fig:
                    img_path = f"{tmpdir}/Map_{i+1}.png"
                    fig.write_image(img_path, width=900, height=600)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, f"Map - {indicators[i]}", ln=True, align="C")
                    pdf.image(img_path, x=10, y=20, w=277)

        elif active_tab == "bar-tab":
            figs = update_bar_plots(hh_type, category, indicators, None, selected_state)
            for i, fig in enumerate(figs):
                if fig:
                    img_path = f"{tmpdir}/Bar_{i+1}.png"
                    fig.write_image(img_path, width=900, height=600)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, f"Bar Chart - {indicators[i]}", ln=True, align="C")
                    pdf.image(img_path, x=10, y=20, w=277)

        elif active_tab == "violin-tab":
            figs = update_violin_plots(hh_type, category, indicators, selected_state)
            for i, fig in enumerate(figs):
                if fig:
                    img_path = f"{tmpdir}/Violin_{i+1}.png"
                    fig.write_image(img_path, width=800, height=500)
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, f"Violin Plot - {indicators[i]}", ln=True, align="C")
                    pdf.image(img_path, x=10, y=20, w=277)

        elif active_tab == "bubble-tab":
            fig = update_combined_bubble(hh_type, category, indicators, selected_state)
            if fig:
                img_path = f"{tmpdir}/Bubble_Plot.png"
                fig.write_image(img_path, width=900, height=500)

                legend_text = f"""X-Axis: {indicators[0]}\nY-Axis: {indicators[1]}\nSize: {indicators[2]}\nColor: {indicators[3]}"""

                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(0, 10, "Bubble Plot", ln=True, align="C")
                pdf.set_font("Arial", "", 10)
                for line in legend_text.split("\n"):
                    pdf.cell(0, 8, line, ln=True, align="C")

                pdf.image(img_path, x=10, y=50, w=277)

        pdf_bytes = pdf.output(dest="S").encode('latin1')
        return dcc.send_bytes(pdf_bytes, filename="Tribal_Health_Dashboard.pdf")

@app.callback(
    Output('indicator-selectors', 'children'),
    [Input('state-selection', 'value'),
     Input('indicator-category-dropdown', 'value')],
    [State({'type': 'indicator-dropdown', 'index': ALL}, 'value')]
)
def update_indicators(selected_state_acronym, selected_category, existing_values):
    if not selected_category:
        return None

    data_source = state_data[selected_state_acronym] if selected_state_acronym else df_state
    data_source = data_source[data_source['category'] == selected_category]

    indicators = sorted(data_source["indicator_name"].dropna().unique())
    if not existing_values or all(v is None for v in existing_values):
        default_indicators = [
            "Households having access to internet (%)",
            "Households owning a mobile/telephone (%)",
            "Women(15-49 years) with exposure to mass media i.e.,newspaper,TV,radio,watch movie in theatre (%)",
            "Women(15-49 years) using internet (%)"
        ]
        existing_values = default_indicators
    used = set(existing_values or [])
    dropdowns = []

    for i in range(4):
        options = [{'label': ind, 'value': ind}
                   for ind in indicators
                   if ind not in used or ind == (existing_values[i] if existing_values and len(existing_values) > i else None)]
        dropdowns.append(dbc.Col([
            dcc.Dropdown(
                id={'type': 'indicator-dropdown', 'index': i+1},
                options=options,
                value=existing_values[i] if existing_values and len(existing_values) > i else None,
                style={'fontSize': '12px'},
                optionHeight=50,
                placeholder=f"Select Indicator {i+1}"
            )
        ], width=3))
        if existing_values and len(existing_values) > i:
            used.add(existing_values[i])

    return dbc.Row(dropdowns, justify="center", className="mb-3")

@app.callback(
    Output('state-selection', 'value'),
    Input('reset', 'n_clicks'),
    prevent_initial_call=True
)
def reset_to_country(n_clicks):
    return None

@app.callback(
    Output('selected-state-name', 'children'),
     Input('state-selection', 'value')
)
def update_title(selected_state):
    if selected_state:
        return f"District-level View: {state_name_mapping.get(selected_state, selected_state)}"
    return "India-level View"

@app.callback(
    Output({'type': 'indicator-dropdown', 'index': ALL}, 'options'),
    Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
    State('indicator-category-dropdown', 'value'),
    State('state-selection', 'value')
)
def update_indicator_options(selected_values, selected_category, selected_state):
    data_source = state_data[selected_state] if selected_state else df_state
    if selected_category:
        data_source = data_source[data_source['category'] == selected_category]

    all_indicators = sorted(data_source['indicator_name'].dropna().unique())
    options_per_dropdown = []

    for i in range(4):
        used_elsewhere = set(val for j, val in enumerate(selected_values) if j != i and val)
        options = [
            {'label': ind, 'value': ind}
            for ind in all_indicators
            if ind not in used_elsewhere or ind == selected_values[i]
        ]
        options_per_dropdown.append(options)
    return options_per_dropdown

@app.callback(
    [Output('violin-1', 'figure'), Output('violin-2', 'figure'),
     Output('violin-3', 'figure'), Output('violin-4', 'figure')],
    Input('category-selection-type', 'value'),
    Input('indicator-category-dropdown', 'value'),
    Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
    Input('state-selection', 'value')
)
def update_violin_plots(hh_type, category, indicators, selected_state):
    df = df_district[df_district['state_acronym'] == selected_state] if selected_state else df_state
    if category:
        df = df[df['category'] == category]

    df['ST'] = pd.to_numeric(df['ST'], errors='coerce')
    df['Non-ST'] = pd.to_numeric(df['Non-ST'], errors='coerce')
    df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

    def generate_violin(indicator):
        if not indicator:
            return px.scatter(title="No Indicator Selected")

        sub_df = df[df['indicator_name'] == indicator].copy()
        if sub_df.empty:
            return px.scatter(title=f"{indicator} - No Data")

        sub_df['x_dummy'] = ''
        label_col = 'district_id' if selected_state else 'state_name'
        indicator_type = indicator_type_map.get(indicator, "Neutral")

        colors = []
        for val in sub_df[hh_type]:
            if pd.isna(val):
                colors.append('gray')
            elif indicator_type == 'Positive':
                if val >= 75:
                    colors.append('green')
                elif val >= 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            elif indicator_type == 'Negative':
                if val <= 25:
                    colors.append('green')
                elif val <= 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            else:
                colors.append('gray')

        violin = go.Violin(
            y=sub_df[hh_type],
            x=sub_df['x_dummy'],
            box_visible=True,
            line_color='blue',
            fillcolor='rgba(0,0,255,0.1)',
            points=False,
            name='Distribution'
        )

        scatter = go.Scatter(
            x=sub_df['x_dummy'],
            y=sub_df[hh_type],
            mode='markers',
            marker=dict(color=colors, size=7, line=dict(width=0.5, color='black')),
            customdata=sub_df[label_col],
            hovertemplate="%{customdata}<br>Value: %{y:.1f}<extra></extra>",
            name='Points'
        )

        fig = go.Figure(data=[violin, scatter])
        fig.update_layout(
            title=None,
            xaxis_title=None,
            yaxis_title=hh_type,
            margin=dict(t=50, b=50, l=40, r=40),
            showlegend=False
        )
        return fig

    indicators = indicators + [None] * (4 - len(indicators))
    return (
        generate_violin(indicators[0]),
        generate_violin(indicators[1]),
        generate_violin(indicators[2]),
        generate_violin(indicators[3]),
    )

@app.callback(
    Output('bubble-plot', 'figure'),
    Input('category-selection-type', 'value'),
    Input('indicator-category-dropdown', 'value'),
    Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
    Input('state-selection', 'value')
)
def update_combined_bubble(hh_type, category, indicators, selected_state):
    indicators = [i for i in indicators if i]
    if len(indicators) < 4:
        return px.scatter(title="Please select 4 indicators for bubble chart.")

    ind_x, ind_y, ind_size, ind_color = indicators[:4]

    use_district = selected_state is not None
    df = df_district[df_district['state_acronym'] == selected_state] if use_district else df_state
    if category:
        df = df[df['category'] == category]

    df = df[df['indicator_name'].isin([ind_x, ind_y, ind_size, ind_color])]
    df[hh_type] = pd.to_numeric(df[hh_type], errors='coerce')
    label_col = 'district_id' if use_district else 'state_name'
    if label_col not in df.columns:
        return px.scatter(title=f"{label_col} column missing")

    pivot = df.pivot_table(index=label_col, columns='indicator_name', values=hh_type).dropna().reset_index()

    fig = px.scatter(
        pivot, x=ind_x, y=ind_y,
        size=ind_size, color=ind_color,
        hover_name=label_col,
        hover_data={ind_color: True, ind_x: False, ind_y: False, ind_size: False}
    )
    fig.update_traces(hovertemplate="%{hovertext}<br>")
    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(t=60, b=40, l=40, r=40),
        coloraxis_colorbar=dict(title=None),
        showlegend=False
    )
    return fig

@app.callback(
    [Output({'type': 'map', 'index': i}, 'figure') for i in range(1, 5)],
    [Input('category-selection-type', 'value'),
     Input('indicator-category-dropdown', 'value'),
     Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
     Input('state-selection', 'value')]
)
def update_all_maps(hh_type, category, indicators, selected_state):
    indicators = indicators + [None] * (4 - len(indicators))
    
    if selected_state:
        # District-level view
        return [generate_district_map(hh_type, category, ind, selected_state) for ind in indicators]
    else:
        # State-level view
        return [generate_state_map(hh_type, category, ind) for ind in indicators]

def generate_state_map(hh_type, category, indicator):
    if not indicator:
        return px.scatter(title="No Indicator Selected")

    # Filter the state-level data
    df = df_state.copy()
    if category:
        df = df[df['category'] == category]
    df = df[df['indicator_name'] == indicator]
    df[hh_type] = pd.to_numeric(df[hh_type], errors='coerce')
    df["state_acronym"] = df["state_acronym"].astype(str)

    # All state acronyms from GeoJSON
    geo_ids = [f["properties"]["state_acronym"] for f in geojson_states["features"]]
    all_states = pd.DataFrame({"state_acronym": geo_ids})

    # Merge with data, handle missing values
    merged = pd.merge(all_states, df[["state_acronym", hh_type]], on="state_acronym", how="left")
    merged["plot_value"] = merged[hh_type].fillna(-1)

    # Hover text
    acronym_to_name = {
        f["properties"]["state_acronym"]: f["properties"].get("state_name", f["properties"]["state_acronym"])
        for f in geojson_states["features"]
    }
    merged["hover_text"] = merged["state_acronym"].apply(
        lambda x: f"{acronym_to_name.get(x, x)}<br>Value: {merged.loc[merged['state_acronym'] == x, hh_type].values[0]:.1f}"
        if pd.notna(merged.loc[merged['state_acronym'] == x, hh_type].values[0])
        else f"{acronym_to_name.get(x, x)}<br>Value: NA"
    )

    # Color scale range
    valid_vals = merged["plot_value"].replace(-1, pd.NA).dropna()
    color_range = (valid_vals.min(), valid_vals.max()) if not valid_vals.empty else (0, 100)

    # Map center (India default)
    center_lat, center_lon, zoom_level = 22, 80, 3

    # Create choropleth mapbox
    fig = px.choropleth_mapbox(
        merged,
        geojson=geojson_states,
        locations="state_acronym",
        featureidkey="properties.state_acronym",
        color="plot_value",
        color_continuous_scale=[[0.0, "gray"], [0.01, "#c6dbef"], [1.0, "#084594"]],
        range_color=color_range,
        mapbox_style="carto-positron",
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom_level,
        title=None
    )

    fig.update_traces(
        hovertemplate="%{customdata}<extra></extra>",
        customdata=merged["hover_text"],
        marker_line_color='black',
        marker_line_width=0.5
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=hh_type),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig

def generate_district_map(hh_type, category, indicator, selected_state):
    if not indicator:
        return px.scatter(title="No Indicator Selected")

    # Filter the district data
    df = df_district[df_district['state_acronym'] == selected_state].copy()
    if category:
        df = df[df['category'] == category]
    df = df[df['indicator_name'] == indicator]
    df[hh_type] = pd.to_numeric(df[hh_type], errors='coerce')
    df["district_id"] = df["district_id"].astype(str)

    # Filter GeoJSON to only selected state's districts
    filtered_features = [
        f for f in geojson_districts["features"]
        if f["properties"]["state_acronym"] == selected_state
    ]
    filtered_geojson = {
        "type": "FeatureCollection",
        "features": filtered_features
    }

    # All district IDs for the selected state from GeoJSON
    geo_ids = [str(f["properties"]["district_id"]) for f in filtered_features]
    all_districts = pd.DataFrame({"district_id": geo_ids})

    # Merge data and fill missing values
    merged = pd.merge(all_districts, df[["district_id", hh_type]], on="district_id", how="left")
    merged["plot_value"] = merged[hh_type].fillna(-1)

    # Hover text
    id_to_name = {
        str(f["properties"]["district_id"]): f["properties"].get("district_name", "Unknown")
        for f in filtered_features
    }
    merged["hover_text"] = merged["district_id"].apply(
        lambda x: f"{id_to_name.get(x, 'Unknown')}<br>Value: {merged.loc[merged['district_id'] == x, hh_type].values[0]:.1f}"
        if pd.notna(merged.loc[merged['district_id'] == x, hh_type].values[0])
        else f"{id_to_name.get(x, 'Unknown')}<br>Value: NA"
    )

    # Compute color scale range
    valid_vals = merged["plot_value"].replace(-1, pd.NA).dropna()
    color_range = (valid_vals.min(), valid_vals.max()) if not valid_vals.empty else (0, 100)

    # Set Mapbox center and zoom based on state geometry
    center_lat, center_lon, zoom_level = 22, 80, 3
    for feature in geojson_states["features"]:
        if feature["properties"]["state_acronym"] == selected_state:
            coords = feature["geometry"]["coordinates"]
            flat_coords = [pt for polygon in coords for pt in polygon[0]] if feature["geometry"]["type"] == "MultiPolygon" else coords[0]
            lats = [lat for lon, lat in flat_coords]
            lons = [lon for lon, lat in flat_coords]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            zoom_level = 5
            break

    # Create choropleth mapbox
    fig = px.choropleth_mapbox(
        merged,
        geojson=filtered_geojson,
        locations="district_id",
        featureidkey="properties.district_id",
        color="plot_value",
        color_continuous_scale=[[0.0, "gray"], [0.01, "#c6dbef"], [1.0, "#084594"]],
        range_color=color_range,
        mapbox_style="carto-positron",
        center={"lat": center_lat, "lon": center_lon},
        zoom=zoom_level,
        title=None
    )

    fig.update_traces(
        hovertemplate="%{customdata}<extra></extra>",
        customdata=merged["hover_text"],
        marker_line_color='black',
        marker_line_width=0.5
    )

    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=hh_type),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig

@app.callback(
    [Output('plot-1', 'figure'), Output('plot-2', 'figure'),
     Output('plot-3', 'figure'), Output('plot-4', 'figure')],
    [Input('category-selection-type', 'value'),
     Input('indicator-category-dropdown', 'value'),
     Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
     Input('clicked-state-store', 'data'),
     Input('state-selection', 'value')]
)
def update_bar_plots(selected_data, selected_cat, indicator_values, clicked_state, selected_state):
    def generate_plot(indicator):
        if not indicator:
            return px.scatter(title="No Indicator Selected")

        if selected_state:
            df = df_district[df_district['state_acronym'] == selected_state]
            if df.empty:
                return px.scatter(title=f"{indicator} - No District Data Available")
        else:
            df = df_state

        if selected_cat:
            df = df[df['category'] == selected_cat]
        df = df[df['indicator_name'] == indicator]

        if df.empty:
            return px.scatter(title=f"{indicator} - No Data Available")

        df['ST'] = pd.to_numeric(df['ST'], errors='coerce')
        df['Non-ST'] = pd.to_numeric(df['Non-ST'], errors='coerce')
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce')

        x = df['district_id'] if selected_state else df['state_name']
        bar_values = df[selected_data]
        indicator_type = indicator_type_map.get(indicator, "Neutral")

        colors = []
        for val in bar_values:
            if pd.isna(val):
                colors.append('gray')
            elif indicator_type == 'Positive':
                if val >= 75:
                    colors.append('green')
                elif val >= 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            elif indicator_type == 'Negative':
                if val <= 25:
                    colors.append('green')
                elif val <= 50:
                    colors.append('orange')
                else:
                    colors.append('red')
            else:
                colors.append('gray')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=x,
            y=bar_values,
            marker_color=colors,
            hovertemplate="%{y:.1f}<extra></extra>"
        ))

        fig.update_layout(
            height=350,
            xaxis_tickangle=135,
            xaxis_title=None,
            title=None,
            margin={"t": 50, "b": 50},
            showlegend=False
        )

        return fig

    indicators = indicator_values + [None] * (4 - len(indicator_values))

    return (
        generate_plot(indicators[0]),
        generate_plot(indicators[1]),
        generate_plot(indicators[2]),
        generate_plot(indicators[3]),
    )


@app.callback(
    Output('bubble-legend', 'children'),
    Input({'type': 'indicator-dropdown', 'index': ALL}, 'value')
)
def update_bubble_legend(indicators):
    indicators = indicators + [None] * (4 - len(indicators))
    ind_x, ind_y, ind_size, ind_color = indicators[:4]

    def format_label(name):
        return name if not name else (name[:80] + '...' if len(name) > 80 else name)

    return html.Div([
        html.H6("Legend", className="fw-bold"),
        html.P(f"X-Axis: {format_label(ind_x)}"),
        html.P(f"Y-Axis: {format_label(ind_y)}"),
        html.P(f"Bubble Size: {format_label(ind_size)}"),
        html.P(f"Bubble Color: {format_label(ind_color)}"),
    ])

def get_outlier_summary(df, indicators, hh_type="Total"):
    summary_lines = []
    for ind in indicators:
        sub = df[df['indicator_name'] == ind]
        sub[hh_type] = pd.to_numeric(sub[hh_type], errors='coerce')
        sorted_df = sub.sort_values(by=hh_type)
        low_outlier = sorted_df.iloc[0] if not sorted_df.empty else None
        high_outlier = sorted_df.iloc[-1] if not sorted_df.empty else None

        if low_outlier is not None and pd.notna(low_outlier[hh_type]):
            summary_lines.append(f"Lowest for '{ind}': {low_outlier['state_name']} ({low_outlier[hh_type]:.1f}%)")
        if high_outlier is not None and pd.notna(high_outlier[hh_type]):
            summary_lines.append(f"Highest for '{ind}': {high_outlier['state_name']} ({high_outlier[hh_type]:.1f}%)")
    return "\n\n".join(summary_lines)

def generate_llm_prompt_from_filters(df, selected_state, selected_category, indicators, level):
    if df.empty:
        return "No data available to analyze.", None

    df = df.copy()

    if 'district_id' in df.columns:
        df = df[['state_name', 'district_id', 'indicator_name', 'ST', 'Non-ST', 'Total']]
    else:
        df = df[['state_name', 'indicator_name', 'ST', 'Non-ST', 'Total']]

    location = f"districts within {selected_state}" if level == "district" else "states across India"
    indicator_text = ", ".join(indicators) if indicators else "selected indicators"
    category_text = f"under the category '{selected_category}'" if selected_category else "across all categories"

    def get_outlier_summary(df, indicators, hh_type="Total"):
        summary_lines = []
        for ind in indicators:
            sub = df[df['indicator_name'] == ind]
            sub[hh_type] = pd.to_numeric(sub[hh_type], errors='coerce')
            sorted_df = sub.sort_values(by=hh_type)
            if not sorted_df.empty:
                low = sorted_df.iloc[0]
                high = sorted_df.iloc[-1]
                if pd.notna(low[hh_type]):
                    summary_lines.append(f"Lowest for '{ind}': {low['state_name']} ({low[hh_type]:.1f}%)")
                if pd.notna(high[hh_type]) and high['state_name'] != low['state_name']:
                    summary_lines.append(f"Highest for '{ind}': {high['state_name']} ({high[hh_type]:.1f}%)")
        return "\n".join(summary_lines)

    outlier_summary = get_outlier_summary(df, indicators)

    prompt = f"""
You are a public health data analyst reviewing NFHS survey data from {location}.

The data contains values for {indicator_text}, {category_text}, disaggregated by ST, Non-ST, and Total populations.

Your task:
- For each indicator, identify states with unusually high or low values (i.e., outliers).
- Clearly name the state and value. Focus on Total unless ST/Non-ST differences are striking.
- Briefly explain why each state is an outlier using only visible trends in the data (e.g., gaps between ST and Non-ST).
- Give one actionable, state-specific policy recommendation per outlier.

Outlier Summary:
{outlier_summary}

Instructions:
- Avoid guessing causes not evident in the data.
- Use plain language. Avoid jargon or fabricated terms.
- Do not summarize the full dataset. Focus only on outliers.
""".strip()

    refine_prompt = (
        "Now refine the insights by explaining possible socio-economic or demographic reasons for the outliers, "
        "and tailor the recommendations more specifically for tribal vs non-tribal populations in those states."
    )

    return prompt, refine_prompt


@app.callback(
    Output("main-tabs", "data"),
    [
        Input("tab-map-tab", "n_clicks"),
        Input("tab-bar-tab", "n_clicks"),
        Input("tab-violin-tab", "n_clicks"),
        Input("tab-bubble-tab", "n_clicks")
    ],
    prevent_initial_call=True
)
def switch_tab(*clicks):
    tab_ids = ["map-tab", "bar-tab", "violin-tab", "bubble-tab"]
    for i, _ in enumerate(clicks):
        if ctx.triggered_id == f"tab-{tab_ids[i]}":
            return tab_ids[i]
    return dash.no_update

@app.callback(
    Output('visualization-panel', 'children'),
    [Input('main-tabs', 'data'),
     Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
     Input('state-selection', 'value'),
     Input('indicator-category-dropdown', 'value'),
     Input('category-selection-type', 'value')]
)
def render_visualization_panel(tab, indicators, selected_state, selected_category, hh_type):
    indicators = indicators + [None] * (4 - len(indicators))
    valid_indicators = [ind for ind in indicators if ind]

    # Select data
    df = df_district[df_district['state_acronym'] == selected_state] if selected_state else df_state
    if selected_category:
        df = df[df['category'] == selected_category]
    df = df[df['indicator_name'].isin(valid_indicators)]
    df[hh_type] = pd.to_numeric(df[hh_type], errors='coerce')

    level = "district" if selected_state else "state"

    # Generate agentic prompts
    base_prompt, follow_up_prompt = generate_llm_prompt_from_filters(
        df=df,
        selected_state=state_name_mapping.get(selected_state, selected_state) if selected_state else None,
        selected_category=selected_category,
        indicators=valid_indicators,
        level=level
    )

    def remove_duplicate_heading(text):
        lines = text.strip().splitlines()
        if len(lines) >= 2 and lines[0].strip().lower() == lines[1].strip().lower():
            return "\n".join(lines[1:]).strip()
        return text.strip()

    def highlight_regions(text):
        state_names = df_state['state_name'].unique().tolist()
        district_names = df_district['district_id'].unique().tolist()
        for name in sorted(state_names + district_names, key=len, reverse=True):
            text = re.sub(fr"\b{re.escape(name)}\b", f"**{name}**", text)
        return text

    if not base_prompt:
        llm_text_cleaned = "No data available to analyze."
        followup_text_cleaned = ""
    else:
        try:
            chat_msgs = [{"role": "user", "content": base_prompt}]
            insights_response = ollama.chat(model="deepseek-r1:1.5b", messages=chat_msgs, options={"temperature": 1.5})
            insights_text = insights_response["message"]["content"]

            if follow_up_prompt:
                chat_msgs += [insights_response["message"], {"role": "user", "content": follow_up_prompt}]
                actions_response = ollama.chat(model="deepseek-r1:1.5b", messages=chat_msgs, options={"temperature": 1.5})
                actions_text = actions_response["message"]["content"]
            else:
                actions_text = ""

        except Exception as e:
            insights_text = f"Error fetching AI Insights: {str(e)}"
            actions_text = ""

        llm_text_cleaned = highlight_regions(remove_duplicate_heading(re.sub(r"<think>.*?</think>", "", insights_text, flags=re.DOTALL)))
        followup_text_cleaned = highlight_regions(remove_duplicate_heading(re.sub(r"<think>.*?</think>", "", actions_text, flags=re.DOTALL)))

    def make_plot(title, fig_id):
        return dbc.Col([
            html.Div(
                title or "No Indicator Selected",
                style={
                    "fontWeight": "bold",
                    "fontSize": "16px",
                    "marginBottom": "4px",
                    "padding": "6px 10px",
                    "backgroundColor": "#f8f9fa",
                    "border": "1px solid #ced4da",
                    "borderRadius": "6px",
                    "whiteSpace": "normal"
                }
            ),
            dcc.Loading(dcc.Graph(id=fig_id, config={"displayModeBar": False}))
        ], md=6)

    plots = []
    if tab == "map-tab":
        plots = [
            dbc.Row([make_plot(indicators[0], {'type': 'map', 'index': 1}),
                     make_plot(indicators[1], {'type': 'map', 'index': 2})]),
            dbc.Row([make_plot(indicators[2], {'type': 'map', 'index': 3}),
                     make_plot(indicators[3], {'type': 'map', 'index': 4})])
        ]
    elif tab == "bar-tab":
        plots = [
            dbc.Row([make_plot(indicators[0], 'plot-1'),
                     make_plot(indicators[1], 'plot-2')]),
            dbc.Row([make_plot(indicators[2], 'plot-3'),
                     make_plot(indicators[3], 'plot-4')])
        ]
    elif tab == "violin-tab":
        plots = [
            dbc.Row([make_plot(indicators[0], 'violin-1'),
                     make_plot(indicators[1], 'violin-2')]),
            dbc.Row([make_plot(indicators[2], 'violin-3'),
                     make_plot(indicators[3], 'violin-4')])
        ]
    elif tab == "bubble-tab":
        plots = [dbc.Row([
            dbc.Col([
                html.Div("Bubble Plot", style={
                    "fontWeight": "bold",
                    "fontSize": "16px",
                    "marginBottom": "4px",
                    "padding": "6px 10px",
                    "backgroundColor": "#f8f9fa",
                    "border": "1px solid #ced4da",
                    "borderRadius": "6px",
                    "whiteSpace": "normal"
                }),
                dcc.Loading(dcc.Graph(id='bubble-plot', config={"displayModeBar": False}))
            ], md=9),
            dbc.Col(html.Div(id='bubble-legend'), md=3)
        ])]

    return html.Div([
        *plots,
        html.Hr(),
        html.Div([
            html.H4("AI Insights (Initial)", style={"marginTop": "30px"}),
            dcc.Markdown(llm_text_cleaned, style={"whiteSpace": "pre-wrap"}),
            html.Br(),
            html.H4("AI Insights (Refined)", style={"marginTop": "20px"}),
            dcc.Markdown(followup_text_cleaned, style={"whiteSpace": "pre-wrap"})
        ])
    ])

if __name__ == '__main__':
    app.run(debug=True)
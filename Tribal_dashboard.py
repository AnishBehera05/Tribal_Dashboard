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

state_csv = "State_Level_Tribal_Health_Factsheet_India_NFHS_V.csv"
district_csv = "District_Level_Tribal_Health_Factsheet_India_NFHS_V.csv"
df_state = pd.read_csv(state_csv, encoding='ISO-8859-1')
df_district = pd.read_csv(district_csv, encoding='ISO-8859-1')

with open("NFHS5_statefiles.geojson", "r") as f:
    geojson_states = json.load(f)
with open("NFHS5_districtlevel.geojson", "r") as f:
    geojson_districts = json.load(f)

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
                    dbc.Button("⬅ Back to State View", id="back-button", color="primary", className="me-2"),
                    dbc.Button("⬅ Back to Country View", id="back-country-button", color="primary")
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
                    dbc.Button("📥 Download Plots", id="download-btn", color="success")
                ], className="d-flex justify-content-end"),
                width=6
            )
        ], justify="between", className="mb-3"),

        html.Div(id='indicator-selectors'),

        html.H3(id='selected-state-name', style={'display': 'none'}),

        dcc.Store(id='clicked-state-store'),
        dcc.Download(id="download-figures"),
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
    State('clicked-state-store', 'data'),
    State('state-selection', 'value'),
    prevent_initial_call=True
)
def download_all_figures(n_clicks, active_tab, hh_type, category, indicators, clicked_state, selected_state):
    indicators = indicators + [None] * (4 - len(indicators))
    grouped_figs = {}

    grouped_figs["Maps"] = list(update_all_maps(hh_type, category, indicators, clicked_state, selected_state))
    grouped_figs["Bar Charts"] = list(update_bar_plots(hh_type, category, indicators, clicked_state, selected_state))
    grouped_figs["Violin Plots"] = list(update_violin_plots(hh_type, category, indicators, selected_state))
    grouped_figs["Bubble Plot"] = [update_combined_bubble(hh_type, category, indicators, clicked_state)]

    pdf = FPDF(orientation="L", unit="mm", format="A4")
    width, height = 297, 210

    with tempfile.TemporaryDirectory() as tmpdir:
        for section, figs in grouped_figs.items():
            for i, fig in enumerate(figs):
                if fig:
                    img_path = f"{tmpdir}/{section.replace(' ', '_')}_{i}.png"
                    fig.write_image(img_path, width=900, height=600)

                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    pdf.cell(0, 10, f"{section} - {i+1}", ln=True, align="C")
                    pdf.image(img_path, x=10, y=20, w=277)

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
    Output('clicked-state-store', 'data'),
    [Input({'type': 'map', 'index': ALL}, 'clickData'),
     Input('state-selection', 'value'),
     Input('back-button', 'n_clicks'),
     Input('back-country-button', 'n_clicks')],
    State('clicked-state-store', 'data'),
    prevent_initial_call=True
)
def update_click_store(map_clicks, selected_state, back_clicks, country_clicks, stored_data):
    triggered = ctx.triggered_id

    if triggered in ["back-button", "back-country-button"]:
        return None

    for click in map_clicks:
        if click and 'location' in click['points'][0]:
            loc = click['points'][0]['location']
            if loc in state_data:
                return loc

    if triggered == "state-selection" and stored_data:
        return None

    return stored_data

@app.callback(
    Output('state-selection', 'value'),
    Input('back-country-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_state_selection(n):
    return None

@app.callback(
    Output('selected-state-name', 'children'),
    [Input('clicked-state-store', 'data'),
     Input('state-selection', 'value')]
)
def update_title(clicked_state, selected_state):
    if clicked_state:
        return f"District-level View: {state_name_mapping.get(clicked_state, clicked_state)}"
    elif selected_state:
        return f"State-level View: {state_name_mapping.get(selected_state, selected_state)}"
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
        label_col = 'district_name' if selected_state else 'state_name'
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
    Input('clicked-state-store', 'data')
)
def update_combined_bubble(hh_type, category, indicators, clicked_state):
    indicators = [i for i in indicators if i]
    if len(indicators) < 4:
        return px.scatter(title="Please select 4 indicators for bubble chart.")

    ind_x, ind_y, ind_size, ind_color = indicators[:4]

    use_district = clicked_state is not None
    df = df_district[df_district['state_acronym'] == clicked_state] if use_district else df_state
    if category:
        df = df[df['category'] == category]

    df = df[df['indicator_name'].isin([ind_x, ind_y, ind_size, ind_color])]
    df[hh_type] = pd.to_numeric(df[hh_type], errors='coerce')
    label_col = 'district_name' if use_district else 'state_name'
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
    [Output('map-1', 'figure'), Output('map-2', 'figure'),
     Output('map-3', 'figure'), Output('map-4', 'figure')],
    [Input('category-selection-type', 'value'),
     Input('indicator-category-dropdown', 'value'),
     Input({'type': 'indicator-dropdown', 'index': ALL}, 'value'),
     Input('clicked-state-store', 'data'),
     Input('state-selection', 'value')]
)
def update_all_maps(selected_data, selected_cat, indicator_values, clicked_state, selected_state):
    indicators = indicator_values + [None] * (4 - len(indicator_values))
    use_district = clicked_state is not None
    df = df_district[df_district['state_acronym'] == clicked_state] if use_district else df_state
    geojson = geojson_districts if use_district else geojson_states
    location_col = "district_id" if use_district else "state_acronym"
    feature_key = "properties.district_id" if use_district else "properties.state_acronym"

    def generate_map(indicator):
        if not indicator:
            return px.scatter(title="No Indicator Selected")

        data = df[df["indicator_name"] == indicator]
        data[location_col] = data[location_col].astype(str)
        data[selected_data] = pd.to_numeric(data[selected_data], errors='coerce')

        if data.empty:
            return px.scatter(title=f"{indicator} - No Data")

        fig = px.choropleth(
            data,
            geojson=geojson,
            locations=location_col,
            color=selected_data,
            featureidkey=feature_key,
            color_continuous_scale="Viridis",
            title=indicator
        )

        fig.update_geos(fitbounds="locations", visible=False)
        fig.update_layout(margin={"r": 0, "t": 30, "l": 0, "b": 0}, showlegend=False)
        return fig

    return (
        generate_map(indicators[0]),
        generate_map(indicators[1]),
        generate_map(indicators[2]),
        generate_map(indicators[3]),
    )

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

        x = df['district_name'] if selected_state else df['state_name']

        if selected_data == 'Total':
            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=x,
                y=df['ST'],
                name='ST',
                marker_color='#1f77b4',
                hoverinfo='none'
            ))

            fig.add_trace(go.Bar(
                x=x,
                y=df['Non-ST'],
                name='Non-ST',
                marker_color='#2ca02c',
                hoverinfo='none'
            ))

            fig.update_traces(
                customdata=df['Total'],
                hovertemplate='Total: %{customdata:.1f}<extra></extra>'
            )

            fig.update_layout(
                barmode='stack',
                height=350,
                xaxis_tickangle=45,
                xaxis_title=None,
                title=None,
                margin={"t": 50, "b": 50},
                showlegend=True
            )

        else:
            fig = go.Figure()

            bar_values = df[selected_data]
            indicator_type = indicator_type_map.get(indicator, "Neutral")

            colors = []
            for val in bar_values:
                if pd.isna(val):
                    colors.append('gray')
                    continue

                if indicator_type == 'Positive':
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

            fig.add_trace(go.Bar(
                x=x,
                y=bar_values,
                marker_color=colors,
                hovertemplate="%{y:.1f}<extra></extra>"
            ))

            fig.update_layout(
                height=350,
                xaxis_tickangle=45,
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
        html.P(f"🔹 X-Axis: {format_label(ind_x)}"),
        html.P(f"🔹 Y-Axis: {format_label(ind_y)}"),
        html.P(f"🔹 Bubble Size: {format_label(ind_size)}"),
        html.P(f"🔹 Bubble Color: {format_label(ind_color)}"),
    ])

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
     Input({'type': 'indicator-dropdown', 'index': ALL}, 'value')]
)
def render_visualization_panel(tab, indicators):
    indicators = indicators + [None] * (4 - len(indicators))

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

    if tab == "map-tab":
        return html.Div([
            dbc.Row([make_plot(indicators[0], {'type': 'map', 'index': 1}),
                     make_plot(indicators[1], {'type': 'map', 'index': 2})]),
            dbc.Row([make_plot(indicators[2], {'type': 'map', 'index': 3}),
                     make_plot(indicators[3], {'type': 'map', 'index': 4})])
        ])
    elif tab == "bar-tab":
        return html.Div([
            dbc.Row([make_plot(indicators[0], 'plot-1'),
                     make_plot(indicators[1], 'plot-2')]),
            dbc.Row([make_plot(indicators[2], 'plot-3'),
                     make_plot(indicators[3], 'plot-4')])
        ])
    elif tab == "violin-tab":
        return html.Div([
            dbc.Row([make_plot(indicators[0], 'violin-1'),
                     make_plot(indicators[1], 'violin-2')]),
            dbc.Row([make_plot(indicators[2], 'violin-3'),
                     make_plot(indicators[3], 'violin-4')])
        ])
    elif tab == "bubble-tab":
        return dbc.Row([
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
        ])
    else:
        return html.Div("Unknown tab selected")

if __name__ == '__main__':
    app.run(debug=True)
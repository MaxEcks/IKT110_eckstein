import json
from logging import raiseExceptions
from os import read
import random
import pickle
import base64
from datetime import datetime

import pandas as pd
import numpy as np
import dash
from dash import dcc
from dash import html as html
from dash.dependencies import Input, Output, State
import plotly.express as px

now = datetime.now()
theta = None

try:
    with open("./pickle/theta_price_model.pkl", "rb") as f:
        theta = pickle.load(f)
except:
    raise Exception("Error loading theta from pickle")


def normalize_inputs(xs, param_path="./pickle/parameters.pkl"):
    with open(param_path, "rb") as f:
        params = pickle.load(f)

    normalized_xs = []
    param_keys = list(params.keys())

    for i, key in enumerate(param_keys):
        if i >= len(xs):
            break  # Avoid index error if xs is shorter than param_keys

        value = xs[i]
        if key in params and value is not None:
            mean = params[key]["mean"]
            std = params[key]["std"]
            normalized_value = (value - mean) / std if std != 0 else 0
        else:
            normalized_value = value  # Leave as-is if not in params or None

        normalized_xs.append(normalized_value)

    if len(xs) > len(param_keys):
        normalized_xs.extend(xs[len(param_keys) :])

    return normalized_xs


def big_front_lobe_ai_price_model(input):
    global theta
    if theta is None:
        raise Exception("Theta is empty when running predict")
    bias = theta[0]
    weight = theta[1:]
    return bias + np.dot(input, weight)


app = dash.Dash(__name__)
app.title = "KKD - Real Estate Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Arial", "padding": "40px"},
    children=[
        html.H2("KKD - Real Estate Dashboard", style={"textAlign": "center"}),
        html.Div(
            [
                html.H4("Basic Info"),
                html.Label("Year Built:"),
                dcc.Input(
                    id="year", value="1990", type="number", style={"width": "100%"}
                ),
                html.Br(),
                html.Br(),
                html.Label("Remodeled Year:"),
                dcc.Input(
                    id="remodeled", value="2015", type="number", style={"width": "100%"}
                ),
                html.Br(),
                html.Br(),
                html.Label("House Color:"),
                dcc.Dropdown(
                    ["black", "blue", "gray", "green", "red", "unknown", "white"],
                    "blue",
                    id="color",
                    style={"width": "100%"},
                ),
                html.Br(),
                html.Label("Put to Market in:"),
                dcc.Dropdown(
                    ["jan", "feb", "march", "april", "november"],
                    "november",
                    id="month-to-marked",
                    style={"width": "100%"},
                ),
            ],
            style={"maxWidth": "500px", "margin": "0 auto"},
        ),
        html.Hr(),
        html.Div(
            [
                html.H4("House & Condition"),
                dcc.Input(
                    id="condition_rating",
                    type="number",
                    placeholder="condition Rating (0-10)",
                    style={"width": "100%"},
                ),
                dcc.Input(
                    id="bathrooms",
                    type="number",
                    placeholder="Bathrooms",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Input(
                    id="external_storage_m2",
                    type="number",
                    placeholder="External Storage (m²)",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Input(
                    id="kitchens",
                    type="number",
                    placeholder="Kitchens",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Input(
                    id="lot_w",
                    type="number",
                    placeholder="Lot Width",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Input(
                    id="rooms",
                    type="number",
                    placeholder="Rooms",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Input(
                    id="size",
                    type="number",
                    placeholder="Size (m²)",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
            ],
            style={"maxWidth": "500px", "margin": "0 auto"},
        ),
        html.Hr(),
        html.Div(
            [
                html.H4("Amenities"),
                html.Label("Fireplace:"),
                dcc.Checklist(
                    options=[{"label": "Yes", "value": "1"}],
                    id="fireplace",
                    value=[],
                    style={"marginBottom": "10px"},
                ),
                html.Label("Parking:"),
                dcc.Checklist(
                    options=[{"label": "Yes", "value": 1}], id="parking", value=[]
                ),
            ],
            style={"maxWidth": "500px", "margin": "0 auto"},
        ),
        html.Hr(),
        html.Div(
            [
                html.H4("Location"),
                dcc.Dropdown(
                    id="district_id",
                    options=[
                        {
                            "label": "District f987fd63b94d4a03aafdf4b1a9c71107",
                            "value": "f987fd63b94d4a03aafdf4b1a9c71107",
                        },
                        {
                            "label": "District dc287dadff314814820067ee82c247d7",
                            "value": "dc287dadff314814820067ee82c247d7",
                        },
                        {
                            "label": "District c32fac32b21a44c89c8d818b7c1002d7",
                            "value": "c32fac32b21a44c89c8d818b7c1002d7",
                        },
                        {
                            "label": "District 86ccb52721164b988b2ca672c9d3d9b7",
                            "value": "86ccb52721164b988b2ca672c9d3d9b7",
                        },
                    ],
                    placeholder="Select District",
                    style={"width": "100%", "marginBottom": "10px"},
                ),
                dcc.Input(
                    id="sun_factor",
                    type="number",
                    placeholder="Sun Factor (0-10)",
                    style={"width": "100%"},
                ),
                html.Br(),
                html.Br(),
                dcc.Dropdown(
                    id="school_id",
                    style={"width": "100%"},
                ),
            ],
            style={"maxWidth": "500px", "margin": "0 auto"},
        ),
        html.Hr(),
        html.Div(
            [
                html.Button(
                    "Okay", id="submit-button", n_clicks=0, style={"marginTop": "30px"}
                ),
            ],
            style={"textAlign": "center"},
        ),
        html.Hr(),
        html.Div(
            [
                html.H4("Predicted Price:"),
                html.Pre(
                    id="output-price", style={"fontSize": "20px", "color": "green"}
                ),
            ],
            style={"maxWidth": "500px", "margin": "0 auto"},
        ),
    ],
)

"""
@app.callback(
   Output("school_id", "options"),
   Input("district_id", "value"),
)
def update_school_dropdown(district_id):
   path = "./pickle/district_linkage.pkl"
   with open(path, "rb") as f:
       district_data = pickle.load(f)

   if district_id is not None and district_id in district_data:
       schools = district_data[district_id].get("schools", [])
       return [{"label": f"School {s['id']}", "value": s["id"]} for s in schools]

   return []

@app.callback(
   Output("output-price", "children"),
   Input("submit-button", "n_clicks"),
   State("year", "value"),
   State("school_id", "value"),
   State("remodeled", "value"),
   State("color", "value"),
   State("month-to-marked", "value"),
   State("condition_rating", "value"),
   State("bathrooms", "value"),
   State("external_storage_m2", "value"),
   State("kitchens", "value"),
   State("lot_w", "value"),
   State("rooms", "value"),
   State("size", "value"),
   State("fireplace", "value"),
   State("parking", "value"),
   State("sun_factor", "value"),
   State("district_id", "value"),
)
def predict_price(
   n_clicks,
   year,
   school_id,
   remodeled,
   house_color,
   month_to_marked,
   condition_rating,
   bathrooms,
   external_storage_m2,
   kitchens,
   lot_w,
   rooms,
   size,
   fireplace,
   parking,
   sun_factor,
   district_id,
):
   if n_clicks == 0:
       return ""

   now = datetime.now()

   path = "./pickle/district_linkage.pkl"
   with open(path, "rb") as f:
       district_data = pickle.load(f)

   if district_data is None:
       raise ValueError()

   district_info = district_data.get(
       district_id,
   )
   print(district_info)
   if district_info is None:
       raise ValueError("district info is empty")

   # Boolean inputs
   has_fireplace = 1 if fireplace else 0
   has_parking = 1 if parking else 0

   # One-hot encode color
   color_options = ["black", "blue", "gray", "green", "red", "unknown", "white"]
   color_vector = [1 if house_color == c else 0 for c in color_options]
   (
       color_black,
       color_blue,
       color_gray,
       color_green,
       color_red,
       color_unknown,
       color_white,
   ) = color_vector

   print(f"Selected district_id: {district_id}")
   print(f"Selected school_id: {school_id}")

   school_data = None
   schools_list = district_info.get("schools", [])

   print(f"Number of schools in district: {len(schools_list)}")
   for s in schools_list:
       print(f"Checking school: {s['id']}")
       if s["id"] == school_id:
           school_data = s
           break

   if school_data is None:
       raise ValueError(
           f"School with ID '{school_id}' not found in district '{district_id}'."
       )

   school_capacity = school_data["capacity"]
   school_age = now.year - school_data["built_year"]
   school_rating = school_data["rating"]

   xs = [
       bathrooms,
       condition_rating,
       external_storage_m2,
       kitchens,
       lot_w,
       rooms,
       size,
       5,  # None,  # storage_rating
       school_age,
       now.year - int(year),  # house_age
       now.year - int(remodeled),  # remodel_age
       has_fireplace,
       has_parking,
       district_info.get("crime_rating", None),  # district_crime_rating
       district_info.get(
           "public_transport_rating", None
       ),  # district_public_transport_rating
       school_rating,
       school_capacity,
       sun_factor,
       color_black,
       color_blue,
       color_gray,
       color_green,
       color_red,
       color_unknown,
       color_white,
   ]
   xs = normalize_inputs(xs)

   result = big_front_lobe_ai_price_model(xs)
   with open("./pickle/parameters.pkl", "rb") as f:
       params = pickle.load(f)
   price_mean = params["price"]["mean"]
   price_std = params["price"]["std"]
   unnormalized_price = result * price_std + price_mean
   return f"{unnormalized_price:.2f}"
"""

@app.callback(
    Output("output-price", "children"),
    Input("submit-button", "n_clicks"),
)
def predict_price_test(n_clicks):
    if n_clicks == 0:
        return ""

    # --- Fixed test values (completely ignore user input) ---
    name_test = {
        "year": 2010,
        "remodeled": 2025,
        "color": "white",
        "month_to_marked": "november",
        "condition_rating": 10,
        "bathrooms": 3,
        "external_storage_m2": 20,
        "kitchens": 3,
        "lot_w": 50,
        "rooms": 10,
        "size": 300,
        "fireplace": 1,  # "no"
        "parking": 1,  # "no"
        "sun_factor": 0.9,
        "district_id": "f987fd63b94d4a03aafdf4b1a9c71107",
        "school_id": "8169505b6a3c4799b4b8e0e58565615f",
    }

    # --- Load district info ---
    path = "./pickle/district_linkage.pkl"
    with open(path, "rb") as f:
        district_data = pickle.load(f)

    district_info = district_data.get(name_test["district_id"])
    if district_info is None:
        raise ValueError(f"District info not found for id {name_test['district_id']}")

    now = datetime.now()

    # Boolean features
    has_fireplace = 1 if name_test["fireplace"] else 0
    has_parking = 1 if name_test["parking"] else 0

    # One-hot encode color
    color_options = ["black", "blue", "gray", "green", "red", "unknown", "white"]
    color_vector = [1 if name_test["color"] == c else 0 for c in color_options]
    (
        color_black,
        color_blue,
        color_gray,
        color_green,
        color_red,
        color_unknown,
        color_white,
    ) = color_vector

    # Get school info
    school_data = None
    for s in district_info.get("schools", []):
        if s["id"] == name_test["school_id"]:
            school_data = s
            break

    if school_data is None:
        raise ValueError(
            f"School with ID '{name_test['school_id']}' not found in district '{name_test['district_id']}'."
        )

    school_capacity = school_data["capacity"]
    school_age = now.year - school_data["built_year"]
    school_rating = school_data["rating"]

    # --- Build feature vector ---
    xs = [
        name_test["bathrooms"],
        name_test["condition_rating"],
        name_test["external_storage_m2"],
        name_test["kitchens"],
        name_test["lot_w"],
        name_test["rooms"],
        name_test["size"],
        8,  # storage_rating (from your data)
        school_age,
        now.year - name_test["year"],
        now.year - name_test["remodeled"],
        has_fireplace,
        has_parking,
        district_info.get("crime_rating"),
        district_info.get("public_transport_rating"),
        school_rating,
        school_capacity,
        name_test["sun_factor"],
        color_black,
        color_blue,
        color_gray,
        color_green,
        color_red,
        color_unknown,
        color_white,
    ]

    xs = normalize_inputs(xs)
    result = big_front_lobe_ai_price_model(xs)
    with open("./pickle/parameters.pkl", "rb") as f:
        params = pickle.load(f)
    price_mean = params["price"]["mean"]
    price_std = params["price"]["std"]
    unnormalized_price = result * price_std + price_mean
    return f"{unnormalized_price:.2f}"


if __name__ == "__main__":
    app.run(debug=True)

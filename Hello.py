# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gzip
import json
import math
import pickle
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from bokeh.io import curdoc
from bokeh.layouts import gridplot
from bokeh.models import BoxSelectTool, CustomJS
from bokeh.plotting import figure, show
from plotly.graph_objs import scatter
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from streamlit.logger import get_logger
from streamlit_folium import st_folium
from streamlit_plotly_events import plotly_events
from streamlit_plotly_mapbox_events import plotly_mapbox_events
from torch.utils.data import DataLoader, TensorDataset

# The plot server must be running
# Go to http://localhost:5006/bokeh to view this plot


LOGGER = get_logger(__name__)

# Initialize session state to store selected indexes
if "si" not in st.session_state:
    st.session_state["si"] = []


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.
        **👈 Select a demo from the sidebar** to see some examples
        of what Streamlit can do!
        ### Want to learn more?
        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)
        ### See more complex demos
        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )


def eliminate_duplicate_columns(df):
    # Find duplicate columns
    duplicate_columns = set()
    for i in range(df.shape[1]):
        col1 = df.iloc[:, i]
        for j in range(i + 1, df.shape[1]):
            col2 = df.iloc[:, j]
            if col1.equals(col2):
                duplicate_columns.add(df.columns.values[j])

    # Drop duplicate columns
    df_cleaned = df.drop(columns=duplicate_columns)

    return df_cleaned


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


@st.cache_data
def get_data():
    with gzip.open("data.pkl.gz", "rb") as f:
        df = pickle.load(f)
        return eliminate_duplicate_columns(df)


def entropy(ds):
    vc = ds.value_counts(normalize=True, sort=False)
    return -np.sum(vc * np.log2(vc))


def get_most_values(ds):
    cnt = ds.value_counts()
    if len(cnt) < 50:
        return list(cnt.index)

    ccnt = np.cumsum(cnt) / np.sum(cnt)
    idx = ccnt < 0.8
    return list(cnt[idx].index)


def encode_cols(df):
    for c in df.columns:
        dc = df[c]
        fc = get_most_values(dc)
        idx = ~dc.isin(fc)
        df.loc[idx, c] = "other"
    return df


def compress_data(df):
    from sklearn.decomposition import TruncatedSVD
    from tqdm import tqdm

    res = []
    for c in tqdm(df.columns):
        dc = pd.get_dummies(df[::10][c].astype(str))
        pca = TruncatedSVD(n_components=min(dc.shape[1], 10))
        pca.fit(dc[::10])
        res += [pca.transform(dc)]
    return np.concatenate(res, axis=1)


def convert_to_pytorch(df, device="cpu"):
    return torch.from_numpy(pd.get_dummies(df.astype(str)).values).float().to(device)


def calculate_likelihood(df, idx, eps=3e-2, min_occurrence=10):
    res = {}
    ratio = idx.mean()
    for c in df.columns:
        dc = df[c]
        vc = dc.astype(str).value_counts()
        if len(vc) < 2:
            continue

        cur_entropy = entropy(dc)
        for v in vc.index:
            cur_idx = dc == v
            if cur_idx.sum() < min_occurrence:
                continue

            cur_ratio = idx[cur_idx].mean()

            if (
                np.isnan(cur_ratio)
                or cur_ratio < eps
                or cur_ratio > 1 - eps
                # or cur_entropy < 1.1
                # or cur_entropy > 3
            ):
                continue
            likelihood = cur_ratio / ratio

            # st.write(cur_ratio, cur_entropy, c, v)
            res[f"{c}-{v}"] = (likelihood - 1) * 100
    res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
    return res


# Function to calculate coordinates
def calculate_coordinates(dfc):
    npc = compress_data(dfc)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(npc[::10])

    df_coords = pd.DataFrame(coords, columns=["x", "y"])
    return df_coords


class AutoencoderWithClassifier(nn.Module):
    def __init__(self, input_size, encoding_size, num_layers, exponent=3, device="cpu"):
        super().__init__()
        encoder_layers = []
        decoder_layers = []

        # Calculate the number of neurons in the first hidden layer
        initial_neurons = input_size
        for _ in range(num_layers):
            next_neurons = int(
                initial_neurons / exponent
            )  # Halve the number of neurons
            encoder_layers.append(
                nn.Linear(initial_neurons, next_neurons, device=device)
            )
            encoder_layers.append(nn.GELU())  # Add GELU activation to encoder layers
            decoder_layers.insert(
                0, nn.Linear(next_neurons, initial_neurons, device=device)
            )  # Insert layers in reverse order
            decoder_layers.insert(0, nn.GELU())  # Add GELU activation to encoder layers
            initial_neurons = next_neurons
        encoder_layers.append(nn.Linear(initial_neurons, encoding_size, device=device))
        encoder_layers.append(nn.GELU())  # Add GELU activation to encoder layers
        decoder_layers.insert(
            0, nn.Linear(encoding_size, initial_neurons, device=device)
        )  # Insert layers in reverse order
        decoder_layers.insert(0, nn.GELU())  # Add GELU activation to encoder layers

        # Add encoder and decoder layers to sequential
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        # Classification head
        self.classifier = nn.Linear(input_size, 2, device=device)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        x_class = self.classifier(x)
        return x_decoded, x_class


def train_model(
    data, labels, encoding_size, num_layers, batch_size=32, num_epochs=10, device="cpu"
):
    # Assuming X is your sparse one-hot encoded data matrix (torch tensor)
    # X should have shape (num_samples, num_features)

    # Create a PyTorch Dataset
    dataset = TensorDataset(
        data.to(device), torch.from_numpy(labels.values).long().to(device)
    )  # Assuming labels is a pandas Series

    # Create a PyTorch DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the model, criterion, and optimizer
    model = AutoencoderWithClassifier(
        data.shape[1], encoding_size, num_layers, exponent=3
    )
    model = model.to("mps")
    criterion_autoencoder = nn.KLDivLoss()
    # criterion_classifier = nn.CrossEntropyLoss()  # Assuming you have class labels as integers
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    epoch_num = 0
    for (
        data,
        labels,
    ) in train_loader:  # Assuming train_loader contains batches of data and labels
        optimizer.zero_grad()
        # reconstructions, class_scores = model(data)
        reconstructions = model(data)[0]

        # Compute autoencoder loss
        loss_autoencoder = criterion_autoencoder(reconstructions, data)

        # # Compute classification loss
        # loss_classifier = criterion_classifier(class_scores, labels)

        # Total loss
        # loss = loss_autoencoder + loss_classifier
        loss = loss_autoencoder

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Print average loss for the epoch
        # print(f'Autoencoder Loss: {loss_autoencoder.item():.4f}, Classification Loss: {loss_classifier.item():.4f}')
        print(f"Autoencoder Loss: {loss_autoencoder.item():.4f}")

        epoch_num += 1
        if epoch_num > num_epochs:
            break

    return model


def selection_fn(trace, points, selector):
    idx = points.point_inds
    return idx


@dataclass
class Point:
    lat: float
    lon: float

    @classmethod
    def from_dict(cls, data: Dict) -> "Point":
        if "lat" in data:
            return cls(float(data["lat"]), float(data["lng"]))
        elif "latitude" in data:
            return cls(float(data["latitude"]), float(data["longitude"]))
        else:
            raise NotImplementedError(data.keys())

    def is_close_to(self, other: "Point") -> bool:
        close_lat = self.lat - 0.0001 <= other.lat <= self.lat + 0.0001
        close_lon = self.lon - 0.0001 <= other.lon <= self.lon + 0.0001
        return close_lat and close_lon


@dataclass
class Bounds:
    south_west: Point
    north_east: Point

    def contains_point(self, point: Point) -> bool:
        in_lon = self.south_west.lon <= point.lon <= self.north_east.lon
        in_lat = self.south_west.lat <= point.lat <= self.north_east.lat

        return in_lon and in_lat

    @classmethod
    def from_dict(cls, data: Dict) -> "Bounds":
        return cls(
            Point.from_dict(data["_southWest"]), Point.from_dict(data["_northEast"])
        )


# @selection_fn
def plot_2d_scatter(df, label):
    # Assuming encoded_data contains the encoded data (torch tensor)
    # encoded_data should have shape (num_samples, 3)
    n_class = len(np.unique(label))
    n_components = st.slider(
        "Number of components", 2, n_class - 1, min(n_class - 1, 4)
    )
    lda = LinearDiscriminantAnalysis(n_components=n_components)

    # Plot the 3D scatter plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ind = np.random.choice(len(df), size=1000, replace=True)

    coords = pd.get_dummies(df.iloc[ind])
    pts = lda.fit_transform(coords, label[ind])
    pts = pd.DataFrame(pts, columns=[f"dim{i}" for i in range(n_components)])

    fig = px.scatter_matrix(pts, dimensions=pts.columns, color=label[ind])
    st.plotly_chart(fig, height=400)

    # mapbox.update_layout(mapbox_style="carto-positron")
    # mapbox.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # mapbox_events = plotly_mapbox_events(
    #     mapbox,
    #     click_event=False,
    #     select_event=True,
    #     hover_event=False,
    # )

    # return fig,

    # fig = px.scatter(
    #     x=pts[:, 0],
    #     y=pts[:, 1],
    #     color=label[ind],)
    # st.plotly_chart(fig)
    #
    # # select_event
    # # selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=True)
    #
    return fig, pts, label[ind]
    # return fig, selected_points


def bokeh_demo(data, label):
    # Writes a component similar to st.write()
    # fig = px.scatter(x=data["dim0"], y=data['dim1'])
    # st.plotly_chart(fig, height=400)
    #
    # # selected_points = plotly_events(fig, click_event=False, hover_event=False, select_event=True)
    # selected_points = fig.data[0].selectedpoints
    # print(selected_points)

    # create three normal population samples with different parameters
    x1 = np.random.normal(loc=5.0, size=400) * 100
    y1 = np.random.normal(loc=10.0, size=400) * 10

    x2 = np.random.normal(loc=5.0, size=800) * 50
    y2 = np.random.normal(loc=5.0, size=800) * 10

    x3 = np.random.normal(loc=55.0, size=200) * 10
    y3 = np.random.normal(loc=4.0, size=200) * 10

    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))

    TOOLS = "pan,wheel_zoom,box_select,lasso_select,reset"

    # create the scatter plot
    p = figure(
        tools=TOOLS,
        width=600,
        height=600,
        min_border=10,
        min_border_left=50,
        toolbar_location="above",
        x_axis_location=None,
        y_axis_location=None,
        title="Linked Histograms",
    )
    p.background_fill_color = "#fafafa"
    p.select(BoxSelectTool).select_every_mousemove = True

    # p.select(LassoSelectTool).select_every_mousemove = False
    #
    r = p.scatter(x, y, size=3, color="#3A5785", alpha=0.6)
    st.bokeh_chart(p)

    # curdoc().add_root(layout)
    curdoc().title = "Selection Histogram"
    selected_points = []

    def update(attr, old, new):
        inds = new
        if len(inds) == 0:
            return
        selected_points = inds
        return new

    r.data_source.selected.on_change("indices", update)

    return r, selected_points


def filter_and_preprocess_data(df):
    # Filter and preprocess data
    filt = (
        df[::10]
        .astype(str)
        .apply(lambda x: (2 ** (entropy(x)) / len(df[::10])), axis=0)
    )
    dfc = df.loc[:, filt < 0.1]
    dfc = dfc.drop(
        [
            "first_name",
            "last_name",
            "returned_at",
            "shipped_at",
            "delivered_at",
            "sale_price",
            "status",
        ],
        axis=1,
    )
    dfc = encode_cols(dfc)

    return dfc, df[::10]["status"]


def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    # Get data
    df = get_data()
    device = "mps"

    # Encode columns
    dfc, label = filter_and_preprocess_data(df)
    label_unique = label.unique()
    st.selectbox("Select a label", label_unique)
    min_occurrence = st.slider(
        "Minimum occurrence", min_value=2, max_value=100, value=10
    )
    eps = st.selectbox("Epsilon", [1e-2, 1e-3, 1e-4, 1e-5])

    # data = convert_to_pytorch(dfc, device=device)
    # label = df['status'] == 'Cancelled'
    # model = train_model(data, label, encoding_size=3, num_layers=5, batch_size=2048,num_epochs=100,device=device)
    # coords = model.encoder(data).detach().cpu().numpy()

    # Calculate likelihoods
    likelihoods = calculate_likelihood(
        dfc, label == "Cancelled", eps=eps, min_occurrence=min_occurrence
    )

    # Filter fields with high likelihood of cancellation
    filtered_fields = {k: v for k, v in likelihoods.items() if v > 5 or v < -5}
    sorted_fields = sorted(filtered_fields.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_fields]
    values = [x[1] for x in sorted_fields]
    # https://community.plotly.com/t/hover-display-values-on-multiple-figures/47590

    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        title="Vertical Bar Plot of Sorted Fields",
        labels={"x": "Likelihood %", "y": "Fields"},
    )
    st.plotly_chart(fig)

    # fig_scatter, data, label = plot_2d_scatter(dfc, df['status'])
    fig_scatter, data, label = plot_2d_scatter(dfc, df["status"])

    # f = go.FigureWidget([go.Scatter(x=data[:,0],y=data[:,1], mode='markers')])

    # fig_scatter.on_selection(selection_fn)

    # _, pts = bokeh_demo(data, label)
    # st.write(pts)


# Execute main function
if __name__ == "__main__":
    # run()
    main()

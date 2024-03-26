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
import multiprocessing
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


@st.cache_resource
def likelihood_inner_loop(c, vc, dc, idx, ratio, min_occurrence):
    res = {}
    if len(vc) < 2:
        return res

    for v in vc.index:
        cur_idx = dc == v
        if idx[cur_idx].sum() < min_occurrence:
            continue

        cur_ratio = idx[cur_idx].mean()
        likelihood = cur_ratio / ratio

        # st.write(cur_ratio, cur_entropy, c, v)
        res[f"{c}-{v}"] = (likelihood - 1) * 100
    return res


@st.cache_data
def calculate_likelihood(df, idx, min_occurrence=10):
    res = {}
    ratio = idx.mean()
    # Create a multiprocessing Pool
    pool = multiprocessing.Pool(processes=8)

    results = []
    for c in df.columns:
        dc = df[c]
        vc = dc.astype(str).value_counts()

        # Map the function to the list of arguments
        results.append(
            pool.apply_async(
                likelihood_inner_loop, [*[c, vc, dc, idx, ratio, min_occurrence]]
            )
        )

    # Close the pool to release resources
    pool.close()
    pool.join()

    # Get results from asynchronous calls
    for result in results:
        if result.get() is not None:
            res.update(result.get())

    res = dict(sorted(res.items(), key=lambda x: x[1], reverse=True))
    return res


# Function to calculate coordinates
def calculate_coordinates(dfc):
    npc = compress_data(dfc)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(npc[::10])

    df_coords = pd.DataFrame(coords, columns=["x", "y"])
    return df_coords


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
    return fig
    # return fig, pts, label[ind]
    # return fig, selected_points


@st.cache_data
def filter_and_preprocess_data(df, perc, label_field="Cancelled"):
    gap = 100 // perc
    # Filter and preprocess data
    filt = (
        df[::gap]
        .astype(str)
        .apply(lambda x: (2 ** (entropy(x)) / len(df[::gap])), axis=0)
    )
    dfc = df.loc[::gap, filt < 0.1]
    lbl = df[::gap]["status"] == label_field
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
        errors="ignore",
    )
    dfc = encode_cols(dfc)

    return dfc, lbl


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
    perc = st.selectbox(
        "Select sampling percentage", [1, 2, 5, 10, 20, 50, 100], index=3
    )
    label_select_all = df["status"].unique()
    label_field = st.selectbox("Select a label", label_select_all)
    dfc, label = filter_and_preprocess_data(df, perc, label_field)

    min_occurrence = st.slider(
        "Minimum occurrence", min_value=2, max_value=100, value=10
    )
    min_abs_likelihood = st.slider(
        "Minimum absolute likelihood", min_value=1, max_value=100, value=5
    )

    # data = convert_to_pytorch(dfc, device=device)
    # label = df['status'] == 'Cancelled'
    # model = train_model(data, label, encoding_size=3, num_layers=5, batch_size=2048,num_epochs=100,device=device)
    # coords = model.encoder(data).detach().cpu().numpy()

    # Calculate likelihoods
    st.write(len(dfc))

    likelihoods = calculate_likelihood(
        dfc.astype(str), label, min_occurrence=min_occurrence
    )

    # Filter fields with high likelihood of cancellation
    filtered_fields = {
        k: v
        for k, v in likelihoods.items()
        if v > min_abs_likelihood or v < -min_abs_likelihood
    }
    sorted_fields = sorted(filtered_fields.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_fields]
    values = [x[1] for x in sorted_fields]
    # https://community.plotly.com/t/hover-display-values-on-multiple-figures/47590

    if len(labels) > 0:
        fig = px.bar(
            x=values,
            y=labels,
            orientation="h",
            title="Likelihood of event depending on each Field",
            labels={"x": "Likelihood %", "y": "Fields"},
        )
        st.plotly_chart(fig)


# Execute main function
if __name__ == "__main__":
    # run()
    main()

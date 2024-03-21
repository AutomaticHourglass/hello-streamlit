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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from streamlit.logger import get_logger
from torch.utils.data import DataLoader, TensorDataset

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
        df[c][idx] = "other"
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


def calculate_likelihood(df, idx, eps=3e-2):
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
            cur_ratio = idx[cur_idx].mean()

            if (
                np.isnan(cur_ratio)
                or cur_ratio < eps
                or cur_ratio > 1 - eps
                or cur_entropy < 1.1
                or cur_entropy > 3
            ):
                continue
            likelihood = cur_ratio / ratio

            st.write(cur_ratio, cur_entropy, c, v)
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


# Main function
def main():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to Streamlit! 👋")

    # Get data
    df = get_data()
    device = "mps"

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

    data = convert_to_pytorch(dfc, device=device)
    # label = df['status'] == 'Cancelled'
    # model = train_model(data, label, encoding_size=3, num_layers=5, batch_size=2048,num_epochs=100,device=device)
    # coords = model.encoder(data).detach().cpu().numpy()

    # Calculate likelihoods
    likelihoods = calculate_likelihood(
        dfc[::10], df[::10]["status"] == "Cancelled", eps=1e-2
    )

    # Filter fields with high likelihood of cancellation
    filtered_fields = {k: v for k, v in likelihoods.items() if v > 5 or v < -5}
    sorted_fields = sorted(filtered_fields.items(), key=lambda x: x[1], reverse=True)
    # st.write("Fields with high likelihood of cancellation:", sorted_fields)

    # https://community.plotly.com/t/hover-display-values-on-multiple-figures/47590

    # Extract labels and values from sorted fields
    labels = [x[0] for x in sorted_fields]
    values = [x[1] for x in sorted_fields]

    fig = px.bar(
        x=values,
        y=labels,
        orientation="h",
        title="Vertical Bar Plot of Sorted Fields",
        labels={"x": "Values", "y": "Fields"},
    )
    st.plotly_chart(fig)

    # # Create a vertical bar plot using Plotly
    # fig = go.Figure(go.Bar(
    #     x=values,
    #     y=labels,
    #     orientation='h',
    #     hoverinfo='x',  # Display only x-values on hover
    # ))

    # # Customize the layout
    # fig.update_layout(
    #     title='Vertical Bar Plot of Sorted Fields',
    #     xaxis_title='Values',
    #     yaxis_title='Fields',
    #     yaxis={'categoryorder': 'total ascending'}
    # )

    # # Display the Plotly figure in Streamlit
    # fig_id = st.plotly_chart(fig).id

    # Assuming encoded_data contains the encoded data (torch tensor)
    # encoded_data should have shape (num_samples, 3)
    lda = LinearDiscriminantAnalysis(n_components=3)

    # Plot the 3D scatter plot
    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ind = np.random.choice(len(data), size=1000, replace=True)

    coords = pd.get_dummies(dfc.iloc[ind])
    pts = lda.fit_transform(coords, df["status"][ind])

    # ax.scatter(*pts.T)
    plt.scatter(pts[:, 2], pts[:, 1], c=df["status"][ind] == "Cancelled", alpha=0.3)

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pts[:, 2], y=pts[:, 1], mode="markers", name="Data"))


# Execute main function
if __name__ == "__main__":
    # run()
    main()

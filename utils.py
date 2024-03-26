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

import inspect
import textwrap

import streamlit as st


def show_code(demo):
    """Showing the code of the demo."""
    show_code = st.sidebar.checkbox("Show code", True)
    if show_code:
        # Showing the code of the demo.
        st.markdown("## Code")
        sourcelines, _ = inspect.getsourcelines(demo)
        st.code(textwrap.dedent("".join(sourcelines[1:])))


# Function to handle rectangle selection
def select_rectangle(x1, y1, x2, y2):
    rect = plt.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
    selected_points = [
        (xi, yi) for xi, yi in zip(x, y) if x1 <= xi <= x2 and y1 <= yi <= y2
    ]
    return selected_points


def double_slider():
    # HTML and JavaScript code for custom double slider
    double_slider_script = """
    <div id="slider-range"></div>
    <script>
      $(function() {
        $("#slider-range").slider({
          range: true,
          min: 0,
          max: 100,
          values: [20, 80],
          slide: function(event, ui) {
            $("#min-value").val(ui.values[0]);
            $("#max-value").val(ui.values[1]);
          }
        });
        $("#min-value").val($("#slider-range").slider("values", 0));
        $("#max-value").val($("#slider-range").slider("values", 1));
      });
    </script>
    """

    # Streamlit sidebar widget for displaying and selecting range
    st.sidebar.markdown("# Select Range")
    st.sidebar.write(
        "Selected Range: ", st.slider("", 0, 100, (20, 80), key="double_slider")
    )

    # Inject JavaScript code into the Streamlit app
    st.markdown(double_slider_script, unsafe_allow_html=True)


def mouse_selection():
    # JavaScript code for mouse selection and freezing plot on Shift keydown
    js = """
    <script>
    var points = [];
    var selected_points = [];
    var plotFrozen = false;

    function updateSelectedRange() {
        var x_range = [];
        var y_range = [];
        for (var i = 0; i < selected_points.length; i++) {
            var point = selected_points[i];
            x_range.push(point.x);
            y_range.push(point.y);
        }
        console.log(x_range, y_range);
        // You can perform further actions with the selected range here
    }

    document.addEventListener('keydown', function(event) {
        if (event.key === 'Shift') {
            plotFrozen = true;
            console.log("Plot frozen");
        }
    });

    document.addEventListener('keyup', function(event) {
        if (event.key === 'Shift') {
            plotFrozen = false;
            console.log("Plot unfrozen");
        }
    });

    document.addEventListener('click', function(event) {
        if (plotFrozen) {
            return;
        }
        var rect = event.target.getBoundingClientRect();
        var x = event.clientX - rect.left;
        var y = event.clientY - rect.top;
        var point = { x: x, y: y };
        points.push(point);
        selected_points.push(point);
        console.log(points);
        updateSelectedRange();
    });
    </script>
    """

    # Inject JavaScript code into the Streamlit app
    st.markdown(js, unsafe_allow_html=True)


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

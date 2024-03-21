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

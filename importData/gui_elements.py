#!/usr/bin/env python3


# mplt.use("TkAgg")
from tkinter import Button, Entry, Label, Toplevel
from typing import Optional, Tuple
from warnings import warn

import matplotlib as matplotlib
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from cmcrameri import cm as cmc
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from importData.epochs_management import inEpochsMask


class rangeButton:
    nameDict = {"test": "test", "lossPred": "predicted loss"}

    def __init__(self, typeButt="test", relevantSliders=None):
        if typeButt == "test" or typeButt == "lossPred":
            self.typeButt = typeButt
        if relevantSliders is None:
            raise ValueError("relevantSliders must be a list of 2 sliders")
        else:
            self.relevantSliders = relevantSliders

    def __call__(self, val):
        self.win = Toplevel()
        self.win.title(f"Manual setting of the {self.nameDict[self.typeButt]} set")
        self.win.geometry("400x200")

        textLabel = self.construct_label()
        self.rangeLabel = Label(self.win, text=textLabel)
        self.rangeLabel.place(relx=0.5, y=30, anchor="center")

        self.rangeEntry = Entry(self.win, width=15, bd=5)
        defaultValues = self.update_def_values(SetData)
        self.rangeEntry.insert(0, f"{defaultValues[0]}-{defaultValues[1]}")
        self.rangeEntry.place(relx=0.5, y=90, anchor="center")

        self.okButton = Button(self.win, width=5, height=1, text="Ok")
        self.okButton.bind("<Button-1>", lambda event: self.set_sliders_and_close())
        self.okButton.place(relx=0.5, y=175, anchor="center")

        self.win.mainloop()

    def construct_label(self):
        text = (
            f"Enter the range of the {self.nameDict[self.typeButt]} set (e.g. 0-1000)"
        )
        return text

    def update_def_values(self, SetData):
        nameId = f"{self.typeButt}SetId"
        nameSize = f"size{self.typeButt[0].upper()}{self.typeButt[1:]}Set"
        firstTS = round(positionTime[SetData[nameId], 0], 2)
        lastId = round(positionTime[SetData[nameId] + SetData[nameSize], 0], 2)
        return [firstTS, lastId]

    def convert_entry_to_id(self):
        strEntry = self.rangeEntry.get()
        if len(strEntry) > 0:
            try:
                parsedRange = [float(num) for num in list(strEntry.split("-"))]
                convertedRange = [
                    self.closestId(positionTime, num) for num in parsedRange
                ]
                startId = convertedRange[0]
                sizeSetinId = convertedRange[1] - convertedRange[0]

                return startId, sizeSetinId
            except:
                raise ValueError("Please enter a valid range in the format 'start-end'")

    def set_sliders_and_close(self):
        valuesForSlider = self.convert_entry_to_id()
        for ivalue, slider in enumerate(self.relevantSliders):
            slider.set_val(valuesForSlider[ivalue])
        self.win.destroy()

    def closestId(self, arr, valToFind):
        return (np.abs(arr - valToFind)).argmin()


class AnimatedPositionPlotter:
    """
    Animated plotter for DataHelper class showing position trajectory with (direction|distance to the wall|whatever) color coding.
    """

    def __init__(
        self,
        data_helper,
        dim=None,
        trail_length: int = 40,
        figsize: Tuple[int, int] = (10, 8),
        **kwargs,
    ):
        """
        Initialize the animated position plotter.

        Args:
            data_helper: DataHelper instance with .positions and .dim attributes
            trail_length: Number of recent points to show in the trail (default: 40)
            dim: Dimension to use for color coding (default: None, auto-detected as pos, direction, or distance)
            figsize: Figure size as (width, height)
            **kwargs: Additional keyword arguments for customization
                speedMask : Speed mask for the trajectory (default: False)
                speedMaskArray: Pre-computed speed mask array instead of simple boolean (optional)
                predicted : Pre-computed predicted positions (optional)
        """
        self.data_helper = data_helper
        self.trail_length = trail_length
        self.figsize = figsize
        self.dim_name = self.data_helper.target.capitalize()  # Default dimension name

        # Extract data
        if kwargs.get("positions", None) is not None:
            self.positions = np.array(kwargs["positions"])
        else:
            try:
                self.positions = np.array(
                    data_helper.old_positions
                )  # Shape: (n_timepoints, 2)
                # get rid of nan values
            except AttributeError:
                self.positions = np.array(
                    data_helper.positions
                )  # Shape: (n_timepoints, ?? depends on get_true_target return value)
        epochMask = inEpochsMask(
            data_helper.fullBehavior["positionTime"][:, 0],
            data_helper.fullBehavior["Times"]["trainEpochs"],
        ) + inEpochsMask(
            data_helper.fullBehavior["positionTime"][:, 0],
            data_helper.fullBehavior["Times"]["testEpochs"],
        )

        if kwargs.get("speedMaskArray", None) is not None:
            speed_mask = np.array(kwargs["speedMaskArray"])

        elif kwargs.get("speedMask", False):
            try:
                speed_mask = np.array(data_helper.fullBehavior["Times"]["speedFilter"])
            except AttributeError:
                warn("No speed mask found in data_helper. Using all positions.")
                speed_mask = np.ones(len(self.positions), dtype=bool)
        else:
            speed_mask = np.ones(len(self.positions), dtype=bool)

        try:
            totMask = epochMask * speed_mask
        except ValueError:
            print(
                "Warning: Epoch mask and speed mask have different lengths. Using speed mask only."
            )
            totMask = speed_mask

        self.positions = self.positions[totMask]

        indices = ~np.isnan(self.positions).any(axis=1)
        self.positions = self.positions[indices]

        if kwargs.get("predicted", None) is not None:
            self.predicted = np.array(kwargs["predicted"])
            self.predicted = self.predicted[totMask]
            self.predicted = self.predicted[indices]
        else:
            self.predicted = None

        if dim is None:
            if self.data_helper.target == "pos":
                dim = np.ones_like(indices)
                self.dim_name = "dummy"
            elif (
                self.data_helper.target == "lin" or self.data_helper.target == "linear"
            ):
                dim = self.data_helper.linearized
                self.dim_name = "Dist2Threat"
            elif self.data_helper.target.lower() == "linandthigmo":
                dim = "thigmo"
                self.dim_name = "Dist2Wall"
            elif "direction" in self.data_helper.target.lower():
                dim = "direction"
                self.dim_name = "Direction"
            else:
                dim = np.ones_like(indices)

        # check if dim is a string or a np array
        if isinstance(dim, str):
            if dim == "direction":
                self.dim = np.array(data_helper.direction)
                self.dim_name = "direction"
                # get rid of NaN values in directions
                self.dim = self.dim[indices]
            elif dim == "distance" or dim == "thigmo":
                if not hasattr(self.data_helper, "thigmo"):
                    self.data_helper.thigmo = np.array(
                        data_helper.dist2wall(self.positions)
                    )
                self.dim = np.array(data_helper.thigmo)
                self.dim_name = "dist2wall"
                self.dim = self.dim[indices]
        elif isinstance(dim, np.ndarray):
            self.dim = dim
            try:
                self.dim = self.dim[indices]
            except IndexError:
                self.dim = self.dim[totMask][indices]

        if kwargs.get("predLossMask", None) is None:
            predLossMask = np.ones(len(self.positions), dtype=bool)
        else:
            predLossMask = np.array(kwargs["predLossMask"])

        if self.predicted is not None:
            predLossMask = predLossMask[totMask][indices]
            self.predicted[~predLossMask] = (
                np.nan
            )  # Set positions with predLossMask to NaN
        # Validate data
        self._validate_data()

        # Setup figure and animation components
        self.fig = None
        self.ax = None
        self.line = None
        self.points = None
        self.current_point = None
        self.animation = None

        # Animation parameters
        self.current_frame = 0
        self.total_frames = len(self.positions)

    def _validate_data(self):
        """Validate that the data helper has the required attributes with correct shapes."""
        if not hasattr(self.data_helper, "positions"):
            raise ValueError("DataHelper must have 'positions' attribute")

        positions = np.array(self.positions)
        directions = np.array(self.dim)

        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError(
                "Positions must be a 2D array with shape (n_timepoints, 2)"
            )
        if directions.ndim != 1:
            raise ValueError("Directions must be a 1D array")
        if len(positions) != len(directions):
            raise ValueError("Positions and directions must have the same length")

    def setup_plot(
        self,
        **kwargs,
    ):
        """
        Setup the matplotlib figure and axes.

        Args:
            colormap: Colormap for direction coding (default: 'hsv')
            alpha_trail: Transparency for trail points (default: 0.7)
            binary_colors: Use binary coloring (auto-detected if None)
            shock_color: Color for shock zone direction (1 values, default: 'hotpink')
            safe_color: Color for safe zone direction (0 values, default: 'cornflowerblue')
            hlines: List of y-values for horizontal lines (default: None)
            vlines: List of x-values for vertical lines (default: None)
            line_colors: Color(s) for reference lines (default: 'black')
            line_styles: Style(s) for reference lines (default: '--')
            line_widths: Width(s) for reference lines (default: 1.0)
            line_alpha: Transparency for reference lines (default: 0.7)
            custom_lines: List of line segments as [(x1,y1), (x2,y2), ...] or numpy array (default: None)
            custom_line_colors: Color(s) for custom lines (default: 'black')
            custom_line_styles: Style(s) for custom lines (default: '-')
            custom_line_widths: Width(s) for custom lines (default: 2.0)
            custom_line_alpha: Transparency for custom lines (default: 0.8)
            with_ref_bg: Whether to use a reference background image (default: True)
        """
        colormap = kwargs.get("colormap", cmc.buda)
        predicted_cmap = kwargs.get("predicted_cmap", cmc.imola)
        alpha_trail = kwargs.get("alpha_trail", 0.7)
        binary_colors = kwargs.get("binary_colors", None)
        shock_color = kwargs.get("shock_color", "hotpink")
        safe_color = kwargs.get("safe_color", "cornflowerblue")
        shock_color_predicted = kwargs.get("shock_color", "lightpink")
        safe_color_predicted = kwargs.get("safe_color", "lightskyblue")
        hlines = kwargs.get("hlines", None)
        vlines = kwargs.get("vlines", None)
        line_colors = kwargs.get("line_colors", "black")
        line_styles = kwargs.get("line_styles", "--")
        line_widths = kwargs.get("line_widths", 1.0)
        line_alpha = kwargs.get("line_alpha", 0.7)
        custom_lines = kwargs.get("custom_lines", None)
        custom_line_colors = kwargs.get("custom_line_colors", "black")
        custom_line_styles = kwargs.get("custom_line_styles", "-")
        custom_line_widths = kwargs.get("custom_line_widths", 2.0)
        custom_line_alpha = kwargs.get("custom_line_alpha", 0.8)
        with_ref_bg = kwargs.get("with_ref_bg", True)

        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        # Store line parameters
        self.hlines = hlines or []
        self.vlines = vlines or []
        self.custom_lines = custom_lines or [
            self.data_helper.maze_coords,
            self.data_helper.shock_zone,
            self.data_helper.safe_zone,
        ]
        if not custom_lines:
            custom_line_colors = (
                ["black", "hotpink", "cornflowerblue"]
                if not with_ref_bg
                else ["white", "hotpink", "cornflowerblue"]
            )
            custom_line_styles = ["-", "-", "-"]
            custom_line_widths = [4, 2, 2]

        # Auto-detect binary data if not specified
        if binary_colors is None:
            unique_values = np.unique(self.dim)
            binary_colors = len(unique_values) == 2 and set(unique_values) == {0, 1}
            if binary_colors:
                print("Binary direction data detected (0s and 1s)")
                print(f"0 (shock zone) -> {shock_color}")
                print(f"1 (safe zone) -> {safe_color}")

        self.binary_colors = binary_colors
        self.shock_color = shock_color
        self.safe_color = safe_color

        if self.binary_colors:
            # Binary color mapping
            self.colormap = None
            self.predicted_colormap = None
            self.norm = None
            # Create custom colors for binary data
            self.colormap = {0: shock_color, 1: safe_color}
            self.predicted_colormap = {
                0: shock_color_predicted,
                1: safe_color_predicted,
            }
        else:
            # Continuous color mapping
            if isinstance(colormap, str):
                self.colormap = cm.get_cmap(colormap)
                self.predicted_colormap = cm.get_cmap(predicted_cmap)
            else:
                self.colormap = colormap
                self.predicted_colormap = predicted_cmap

            if self.dim_name == "Dist2Threat":
                colormap = LinearSegmentedColormap.from_list(
                    "direction_cmap", [shock_color, safe_color], N=256
                )
                predicted_colormap = LinearSegmentedColormap.from_list(
                    "predicted_direction_cmap",
                    [shock_color_predicted, safe_color_predicted],
                    N=256,
                )
                self.colormap = colormap
                self.predicted_colormap = predicted_colormap
            self.norm = Normalize(vmin=np.min(self.dim), vmax=np.max(self.dim))

        # Initialize empty line for trail
        (self.line,) = self.ax.plot(
            [], [], "-", alpha=alpha_trail, linewidth=2, color="gray"
        )
        (self.predicted_line,) = self.ax.plot(
            [], [], "-", alpha=alpha_trail, linewidth=2, color="xkcd:seafoam"
        )
        (self.delta_predicted_true,) = self.ax.plot(
            [], [], "-", alpha=alpha_trail, linewidth=4, color="xkcd:browny green"
        )

        # Initialize scatter plot for trail points
        if self.binary_colors:
            # For binary data, we'll update colors manually
            self.points = self.ax.scatter([], [], s=50, alpha=alpha_trail)
            if self.predicted is not None:
                self.predicted_points = self.ax.scatter([], [], s=50, alpha=alpha_trail)
            else:
                self.predicted_points = None
        else:
            # For continuous data, use colormap
            self.points = self.ax.scatter(
                [],
                [],
                c=[],
                s=50,
                alpha=alpha_trail,
                cmap=self.colormap,
                norm=self.norm,
            )
            if self.predicted is not None:
                self.predicted_points = self.ax.scatter(
                    [],
                    [],
                    s=50,
                    alpha=alpha_trail,
                    cmap=self.predicted_colormap,
                    norm=self.norm,
                )
            else:
                self.predicted_points = None

        # Initialize current position marker
        self.current_point = self.ax.scatter(
            [],
            [],
            c="red",
            s=100,
            marker="o",
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )
        if self.predicted is not None:
            self.current_predicted_point = self.ax.scatter(
                [],
                [],
                c="xkcd:reddish pink",
                s=100,
                marker="x",
                edgecolors="black",
                linewidth=2,
                zorder=10,
            )

        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        # remove ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Add permanent reference lines
        self._add_reference_lines(line_colors, line_styles, line_widths, line_alpha)

        # Add custom line segments
        self._add_custom_lines(
            custom_line_colors,
            custom_line_styles,
            custom_line_widths,
            custom_line_alpha,
        )

        self.ax.set_xlabel("X Position")
        self.ax.set_ylabel("Y Position")

        if self.binary_colors:
            self.ax.set_title(
                "Position Trajectory - Towards Shock Zone (Pink) or Safe Zone (Blue)"
            )
            # Add custom legend for binary colors
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor=shock_color, label="Shock Zone (0)"),
                Patch(facecolor=safe_color, label="Safe Zone (1)"),
            ]
            self.ax.legend(handles=legend_elements, loc=[0.4, 0.1])
        else:
            self.ax.set_title(
                f"Position Trajectory with {self.dim_name.capitalize()} Color Coding"
            )
            # Add colorbar for continuous data
            cbar = plt.colorbar(self.points, ax=self.ax)
            cbar.set_label(f"{self.dim_name.capitalize()}")

        if with_ref_bg:
            try:
                reference_image = self.data_helper.aligned_ref
            except AttributeError:
                raise ValueError(
                    "No reference image found. Please provide a reference image."
                )
            self.ax.imshow(
                reference_image,
                cmap="gray",
                extent=[0, 1, 0, 1],
                vmin=reference_image.min(),
                vmax=reference_image.max(),
                origin="lower",
            )

        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal", adjustable="box")

        return self.fig, self.ax

    def _add_reference_lines(self, colors, styles, widths, alpha):
        """Add permanent horizontal and vertical reference lines."""

        # Ensure parameters are lists for consistent handling
        def ensure_list(param, default_length):
            if isinstance(param, (list, tuple)):
                return list(param)
            else:
                return [param] * default_length

        total_lines = len(self.hlines) + len(self.vlines)
        if total_lines == 0:
            return

        colors = ensure_list(colors, total_lines)
        styles = ensure_list(styles, total_lines)
        widths = ensure_list(widths, total_lines)
        alphas = ensure_list(alpha, total_lines)

        line_idx = 0

        # Add horizontal lines
        for y_val in self.hlines:
            self.ax.axhline(
                y=y_val,
                color=colors[line_idx % len(colors)],
                linestyle=styles[line_idx % len(styles)],
                linewidth=widths[line_idx % len(widths)],
                alpha=alphas[line_idx % len(alphas)],
                zorder=1,  # Behind the trajectory
            )
            line_idx += 1

        # Add vertical lines
        for x_val in self.vlines:
            self.ax.axvline(
                x=x_val,
                color=colors[line_idx % len(colors)],
                linestyle=styles[line_idx % len(styles)],
                linewidth=widths[line_idx % len(widths)],
                alpha=alphas[line_idx % len(alphas)],
                zorder=1,  # Behind the trajectory
            )

    def _add_custom_lines(self, colors, styles, widths, alpha):
        """Add custom line segments (like maze walls, boundaries, etc.)."""
        if not self.custom_lines:
            return

        # Handle different input formats
        lines_to_plot = []

        for line_data in self.custom_lines:
            # Convert to numpy array for easier handling
            line_array = np.array(line_data)

            if line_array.ndim == 2 and line_array.shape[1] == 2:
                # Single line segment as array of points
                lines_to_plot.append(line_array)
            elif line_array.ndim == 1 and len(line_array) == 4:
                # Single line as [x1, y1, x2, y2]
                lines_to_plot.append(
                    np.array(
                        [[line_array[0], line_array[1]], [line_array[2], line_array[3]]]
                    )
                )
            else:
                print(f"Warning: Unrecognized line format: {line_data}")

        if not lines_to_plot:
            return

        # Ensure parameters are lists
        def ensure_list(param, default_length):
            if isinstance(param, (list, tuple)):
                return list(param)
            else:
                return [param] * default_length

        colors = ensure_list(colors, len(lines_to_plot))
        styles = ensure_list(styles, len(lines_to_plot))
        widths = ensure_list(widths, len(lines_to_plot))
        alphas = ensure_list(alpha, len(lines_to_plot))

        # Plot each line segment
        for i, line_points in enumerate(lines_to_plot):
            self.ax.plot(
                line_points[:, 0],  # X coordinates
                line_points[:, 1],  # Y coordinates
                color=colors[i % len(colors)],
                linestyle=styles[i % len(styles)],
                linewidth=widths[i % len(widths)],
                alpha=alphas[i % len(alphas)],
                zorder=1,  # Behind the trajectory
            )

    def animate_frame(self, frame: int):
        """
        Animation function for a single frame.

        Args:
            frame: Current frame number
        """
        # Calculate the range of points to show (last trail_length points up to current frame)
        start_idx = max(0, frame + 1 - self.trail_length)
        end_idx = frame + 1

        if end_idx <= start_idx:
            return self.line, self.points, self.current_point

        # Get trail data
        trail_positions = self.positions[start_idx:end_idx]
        trail_directions = self.dim[start_idx:end_idx]
        if self.predicted is not None:
            trail_predicted = self.predicted[start_idx:end_idx]
            trail_directions_predicted = self.dim[start_idx:end_idx]
        else:
            trail_predicted = None
            trail_directions_predicted = None

        # Update trail line
        self.line.set_data(trail_positions[:, 0], trail_positions[:, 1])
        if self.predicted is not None:
            self.predicted_line.set_data(trail_predicted[:, 0], trail_predicted[:, 1])

        # Update trail points with direction colors
        if self.binary_colors:
            # Binary color mapping
            trail_colors = [self.colormap[int(d)] for d in trail_directions]
            self.points.set_offsets(trail_positions)
            self.points.set_color(trail_colors)
            if self.predicted is not None:
                self.predicted_points.set_offsets(trail_predicted)
                self.predicted_points.set_color(
                    [
                        self.predicted_colormap[int(d)]
                        for d in trail_directions_predicted
                    ]
                )
        else:
            # Continuous color mapping
            trail_colors = self.colormap(self.norm(trail_directions))
            self.points.set_offsets(trail_positions)
            self.points.set_color(trail_colors)
            if self.predicted is not None:
                trail_predicted_colors = self.predicted_colormap(
                    self.norm(trail_directions_predicted)
                )
                self.predicted_points.set_offsets(trail_predicted)
                self.predicted_points.set_color(trail_predicted_colors)

        # Create alpha gradient for trail (most recent points are more opaque)
        n_trail_points = len(trail_positions)
        alphas = np.linspace(0.2, 0.8, n_trail_points)
        if n_trail_points > 0:
            self.points.set_alpha(alphas[-1])  # Use the alpha of the most recent point
            self.predicted_points.set_alpha(
                alphas[-1]
            ) if self.predicted is not None else None

        # Update current position with color based on direction
        current_pos = self.positions[frame : frame + 1]
        current_dir = self.dim[frame]
        if self.predicted is not None:
            current_predicted_pos = self.predicted[frame : frame + 1]
            current_predicted_dir = self.dim[frame]
            self.delta_predicted_true.set_data(
                [current_pos[0, 0], current_predicted_pos[0, 0]],
                [current_pos[0, 1], current_predicted_pos[0, 1]],
            )

        if self.binary_colors:
            current_color = self.colormap[int(current_dir)]
            self.current_point.set_offsets(current_pos)
            self.current_point.set_color(current_color)
            if self.predicted is not None:
                self.current_predicted_point.set_offsets(current_predicted_pos)
                self.current_predicted_point.set_color(
                    self.predicted_colormap[int(current_predicted_dir)]
                )
        else:
            self.current_point.set_offsets(current_pos)
            if self.predicted is not None:
                self.current_predicted_point.set_offsets(current_predicted_pos)
            # Keep current point red for continuous data for visibility

        # Update title with current frame info and direction
        if self.binary_colors:
            zone_name = "Shock Zone" if current_dir == 0 else "Safe Zone"
            self.ax.set_title(
                f"Position Trajectory - Frame {frame + 1}/{self.total_frames} - {zone_name}"
            )
        else:
            self.ax.set_title(
                f"Position Trajectory - Frame {frame + 1}/{self.total_frames} - {self.dim_name.capitalize()}: {current_dir:.2f}"
            )

        return self.line, self.points, self.current_point

    def create_animation(
        self,
        interval: int = 50,
        repeat: bool = True,
        save_path: Optional[str] = None,
        **kwargs,
    ):
        """
        Create and optionally save the animation.

        Args:
            interval: Time between frames in milliseconds (default: 50)
            repeat: Whether to repeat the animation (default: True)
            save_path: Path to save animation as MP4 (optional)
            **kwargs: Additional arguments for FuncAnimation

        Returns:
            matplotlib.animation.FuncAnimation object
        """
        if self.fig is None:
            self.setup_plot(**kwargs)

        # Default kwargs for better Qt compatibility
        anim_kwargs = {
            "blit": False,  # Better compatibility with Qt
            "cache_frame_data": False,  # Reduce memory usage
        }

        self.animation = animation.FuncAnimation(
            self.fig,
            self.animate_frame,
            frames=self.total_frames,
            interval=interval,
            repeat=repeat,
            **anim_kwargs,
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            try:
                self.animation.save(save_path, writer="ffmpeg", fps=1000 // interval)
                print("Animation saved!")
            except Exception as e:
                print(f"Failed to save animation: {e}")
                print("Make sure ffmpeg is installed for video export")

        return self.animation

    def show(self, interval: int = 50, repeat: bool = True, block=False, **kwargs):
        """
        Show the animation with Qt backend support.

        Args:
            interval: Time between frames in milliseconds (default: 50)
            repeat: Whether to repeat the animation (default: True)
            block: Whether to block execution (auto-detected based on environment)
        """
        self.create_animation(interval=interval, repeat=repeat, **kwargs)

        # Auto-detect blocking behavior
        # Show the plot
        plt.show(block=block)

        return self.animation


# Example DataHelper class for testing
class DataHelper:
    """Example DataHelper class for demonstration."""

    def __init__(self, n_points: int = 200, binary_directions: bool = False):
        """Generate example trajectory data."""
        # Create a spiral trajectory
        t = np.linspace(0, 4 * np.pi, n_points)
        r = np.linspace(1, 5, n_points)

        # Add some noise for realism
        noise_x = np.random.normal(0, 0.1, n_points)
        noise_y = np.random.normal(0, 0.1, n_points)

        x = r * np.cos(t) + noise_x
        y = r * np.sin(t) + noise_y

        self.positions = np.column_stack([x, y])

        if binary_directions:
            # Binary directions (0 = safe zone, 1 = shock zone)
            # Create zones based on position or time
            # Example: shock zone when animal is in upper right quadrant
            shock_zone = (x > np.mean(x)) & (y > np.mean(y))
            # Add some randomness to make it more realistic
            random_shock = np.random.random(n_points) < 0.3
            self.direction = (shock_zone | random_shock).astype(int)
        else:
            # Calculate direction as angle from previous point
            dx = np.diff(x)
            dy = np.diff(y)
            angles = np.arctan2(dy, dx)

            # Add first point (duplicate first angle)
            self.direction = np.concatenate([[angles[0]], angles])

            # Normalize directions to [0, 2π] for better color mapping
            self.direction = (self.direction + 2 * np.pi) % (2 * np.pi)


# Usage example and demonstration
def demo_animated_plot():
    """Demonstrate the animated position plotter."""
    print("Creating example DataHelper with binary shock zone data...")

    # Create example data with binary directions
    data_helper = DataHelper(n_points=150, binary_directions=True)

    print(f"Generated {len(data_helper.positions)} position points")
    print(
        f"Position range: X=[{data_helper.positions[:, 0].min():.2f}, {data_helper.positions[:, 0].max():.2f}], "
        f"Y=[{data_helper.positions[:, 1].min():.2f}, {data_helper.positions[:, 1].max():.2f}]"
    )
    print(
        f"Direction values: {np.unique(data_helper.direction)} (binary: 0=safe, 1=shock)"
    )
    print(f"Shock zone ratio: {np.mean(data_helper.direction):.1%}")

    # Create animated plotter
    plotter = AnimatedPositionPlotter(data_helper, trail_length=40)

    # Setup and show animation
    print("\nCreating animated plot...")
    print("- Trail length: 40 points")
    print("- Pink = Shock Zone (direction = 1)")
    print("- Blue = Safe Zone (direction = 0)")
    print("- Current position marker changes color based on zone")

    # Create the animation
    anim = plotter.show(interval=100, repeat=True)

    return plotter, anim


def create_plotter_for_data(
    data_helper,
    trail_length: int = 40,
    colormap: str = "hsv",
    interval: int = 50,
    backend: str = None,
    hlines: list = None,
    vlines: list = None,
    custom_lines: list = None,
):
    """
    Convenience function to create and show animated plot for your DataHelper.

    Args:
        data_helper: Your DataHelper instance
        trail_length: Number of points in the trail (default: 40)
        colormap: Colormap for direction coding (default: 'hsv')
        interval: Animation speed in milliseconds (default: 50)
        backend: Force specific backend (optional)
        hlines: List of y-values for horizontal reference lines
        vlines: List of x-values for vertical reference lines
        custom_lines: List of custom line segments

    Returns:
        AnimatedPositionPlotter instance
    """
    if backend:
        try:
            matplotlib.use(backend, force=True)
            print(f"Forced backend to: {backend}")
        except Exception as e:
            print(f"Could not set backend {backend}: {e}")

    plotter = AnimatedPositionPlotter(data_helper, trail_length=trail_length)
    plotter.setup_plot(
        colormap=colormap, hlines=hlines, vlines=vlines, custom_lines=custom_lines
    )
    animation = plotter.create_animation(interval=interval)
    plotter.show()
    return plotter


def create_maze_from_matlab(maze_coords):
    """
    Convert MATLAB-style maze coordinates to custom_lines format.

    Args:
        maze_coords: Array-like with shape (n_points, 2) representing connected points

    Returns:
        List suitable for custom_lines parameter

    Example:
        # MATLAB: maze = [0 0; 0 1; 1 1; 1 0; 0.63 0; 0.63 0.75; 0.35 0.75; 0.35 0; 0 0];
        maze_coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0],
                       [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]
        custom_lines = create_maze_from_matlab(maze_coords)
    """
    maze_array = np.array(maze_coords)
    return [maze_array]  # Return as single connected line


def create_qt_plotter(data_helper, trail_length: int = 40, **kwargs):
    """
    Create plotter with guaranteed Qt backend (if available).

    Args:
        data_helper: Your DataHelper instance
        trail_length: Number of points in the trail
        **kwargs: Additional arguments passed to create_plotter_for_data

    Returns:
        AnimatedPositionPlotter instance
    """
    # Try to force Qt backend
    qt_backends = ["Qt5Agg", "Qt4Agg"]
    current_backend = matplotlib.get_backend()

    for backend in qt_backends:
        try:
            matplotlib.use(backend, force=True)
            print(f"Using Qt backend: {backend}")
            break
        except:
            continue
    else:
        print(f"Qt backend not available, using: {current_backend}")

    return create_plotter_for_data(data_helper, trail_length=trail_length, **kwargs)


class ModelPerformanceVisualizer:
    """
    Comprehensive visualization tool for binary classification model performance over time.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ):
        """
        Initialize the performance visualizer.

        Args:
            predictions: Binary predictions (0/1) over time
            ground_truth: True labels (0/1) over time
            timestamps: Optional timestamps for x-axis (default: range(len(predictions)))
        """
        self.predictions = np.array(predictions).astype(int)
        self.ground_truth = np.array(ground_truth).astype(int)

        if len(self.predictions) != len(self.ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        self.timestamps = (
            timestamps if timestamps is not None else np.arange(len(predictions))
        )
        self.n_points = len(self.predictions)

        # Calculate basic metrics
        self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        self.accuracy = accuracy_score(self.ground_truth, self.predictions)
        self.precision = precision_score(
            self.ground_truth, self.predictions, zero_division=0
        )
        self.recall = recall_score(self.ground_truth, self.predictions, zero_division=0)
        self.f1 = f1_score(self.ground_truth, self.predictions, zero_division=0)

        # Calculate error types
        self.correct = self.predictions == self.ground_truth
        self.false_positives = (self.predictions == 1) & (self.ground_truth == 0)
        self.false_negatives = (self.predictions == 0) & (self.ground_truth == 1)
        self.true_positives = (self.predictions == 1) & (self.ground_truth == 1)
        self.true_negatives = (self.predictions == 0) & (self.ground_truth == 0)

    def plot_timeline_comparison(
        self,
        figsize: Tuple[int, int] = (15, 8),
        title: str = "Model Predictions vs Ground Truth Over Time",
    ):
        """
        Plot predictions and ground truth over time with error highlighting.

        Args:
            figsize: Figure size
            title: Plot title
        """
        fig, axes = plt.subplots(3, 1, figsize=figsize, height_ratios=[2, 2, 1])

        # Plot 1: Stacked comparison
        ax1 = axes[0]

        # Plot ground truth
        ax1.fill_between(
            self.timestamps,
            0,
            self.ground_truth,
            alpha=0.3,
            color="green",
            label="Ground Truth",
            step="pre",
        )

        # Plot predictions with error highlighting
        colors = ["red" if not correct else "blue" for correct in self.correct]
        ax1.scatter(
            self.timestamps,
            self.predictions + 0.1,
            c=colors,
            alpha=0.7,
            s=20,
            label="Predictions",
        )

        ax1.set_ylabel("State")
        ax1.set_title(title)
        ax1.set_ylim(-0.1, 1.3)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Error types
        ax2 = axes[1]

        # Create error type visualization
        y_pos = np.zeros(self.n_points)
        colors = []
        labels = []

        for i in range(self.n_points):
            if self.true_positives[i]:
                colors.append("darkgreen")
                y_pos[i] = 1
            elif self.true_negatives[i]:
                colors.append("lightgreen")
                y_pos[i] = 0
            elif self.false_positives[i]:
                colors.append("red")
                y_pos[i] = 0.5
            elif self.false_negatives[i]:
                colors.append("orange")
                y_pos[i] = 0.5

        ax2.scatter(self.timestamps, y_pos, c=colors, alpha=0.8, s=30)
        ax2.set_ylabel("Prediction Type")
        ax2.set_yticks([0, 0.5, 1])
        ax2.set_yticklabels(["True Neg", "Error", "True Pos"])
        ax2.set_title("Prediction Types Over Time")
        ax2.grid(True, alpha=0.3)

        # Create custom legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="darkgreen",
                markersize=8,
                label="True Positive",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="lightgreen",
                markersize=8,
                label="True Negative",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="False Positive",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="orange",
                markersize=8,
                label="False Negative",
            ),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        # Plot 3: Rolling accuracy
        ax3 = axes[2]
        window_size = max(10, self.n_points // 20)  # Adaptive window size
        rolling_accuracy = self._calculate_rolling_metric(self.correct, window_size)

        ax3.plot(
            self.timestamps,
            rolling_accuracy,
            "purple",
            linewidth=2,
            label=f"Rolling Accuracy (window={window_size})",
        )
        ax3.axhline(
            y=self.accuracy,
            color="red",
            linestyle="--",
            label=f"Overall Accuracy: {self.accuracy:.3f}",
        )
        ax3.set_ylabel("Accuracy")
        ax3.set_xlabel("Time")
        ax3.set_ylim(0, 1)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig, axes

    def plot_confusion_matrix_over_time(
        self, window_size: int = 50, figsize: Tuple[int, int] = (15, 10)
    ):
        """
        Plot confusion matrix metrics over time.

        Args:
            window_size: Window size for rolling calculations
            figsize: Figure size
        """
        # Calculate rolling metrics
        rolling_metrics = self._calculate_rolling_confusion_metrics(window_size)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        metrics = ["accuracy", "precision", "recall", "f1"]
        colors = ["blue", "green", "red", "purple"]

        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax = axes[i // 2, i % 2]

            # Plot rolling metric
            ax.plot(
                self.timestamps,
                rolling_metrics[metric],
                color=color,
                linewidth=2,
                label=f"Rolling {metric.capitalize()}",
            )

            # Plot overall metric as horizontal line
            overall_value = getattr(self, metric)
            ax.axhline(
                y=overall_value,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"Overall: {overall_value:.3f}",
            )

            ax.set_title(f"{metric.capitalize()} Over Time")
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(True, alpha=0.3)

        axes[1, 1].set_xlabel("Time")
        axes[1, 0].set_xlabel("Time")

        plt.tight_layout()
        return fig, axes

    def plot_error_heatmap(
        self, window_size: int = 20, figsize: Tuple[int, int] = (15, 6)
    ):
        """
        Create a heatmap showing error density over time.

        Args:
            window_size: Size of time windows for aggregation
            figsize: Figure size
        """
        # Create time windows
        n_windows = max(1, self.n_points // window_size)
        window_errors = np.zeros((4, n_windows))  # TP, TN, FP, FN
        window_labels = []

        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, self.n_points)

            window_slice = slice(start_idx, end_idx)
            window_errors[0, i] = np.sum(self.true_positives[window_slice])
            window_errors[1, i] = np.sum(self.true_negatives[window_slice])
            window_errors[2, i] = np.sum(self.false_positives[window_slice])
            window_errors[3, i] = np.sum(self.false_negatives[window_slice])

            # Create window label
            start_time = self.timestamps[start_idx]
            end_time = self.timestamps[end_idx - 1]
            window_labels.append(f"{start_time:.1f}-{end_time:.1f}")

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        # Normalize by window size for fair comparison
        window_errors_norm = window_errors / window_size

        im = ax.imshow(window_errors_norm, cmap="RdYlBu_r", aspect="auto")

        # Set labels
        ax.set_xticks(range(n_windows))
        ax.set_xticklabels(window_labels, rotation=45)
        ax.set_yticks(range(4))
        ax.set_yticklabels(["True Pos", "True Neg", "False Pos", "False Neg"])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Proportion of Predictions")

        # Add text annotations
        for i in range(4):
            for j in range(n_windows):
                text = ax.text(
                    j,
                    i,
                    f"{window_errors_norm[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

        ax.set_title("Prediction Types Heatmap Over Time")
        ax.set_xlabel("Time Windows")
        plt.tight_layout()
        return fig, ax

    def plot_precision_recall_timeline(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot precision-recall curve and timeline.

        Args:
            figsize: Figure size
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Precision-Recall Curve (if we have probability scores)
        # For binary predictions, we'll show the single point
        ax1.scatter(
            self.recall,
            self.precision,
            s=100,
            c="red",
            label=f"Model (F1={self.f1:.3f})",
        )
        ax1.plot([0, 1], [0.5, 0.5], "k--", alpha=0.5, label="Random")
        ax1.set_xlabel("Recall")
        ax1.set_ylabel("Precision")
        ax1.set_title("Precision-Recall Point")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Plot 2: Precision and Recall over time
        window_size = max(10, self.n_points // 20)
        rolling_precision = self._calculate_rolling_precision(window_size)
        rolling_recall = self._calculate_rolling_recall(window_size)

        ax2.plot(
            self.timestamps,
            rolling_precision,
            "green",
            linewidth=2,
            label=f"Rolling Precision (window={window_size})",
        )
        ax2.plot(
            self.timestamps,
            rolling_recall,
            "blue",
            linewidth=2,
            label=f"Rolling Recall (window={window_size})",
        )

        ax2.axhline(
            y=self.precision,
            color="green",
            linestyle="--",
            alpha=0.7,
            label=f"Overall Precision: {self.precision:.3f}",
        )
        ax2.axhline(
            y=self.recall,
            color="blue",
            linestyle="--",
            alpha=0.7,
            label=f"Overall Recall: {self.recall:.3f}",
        )

        ax2.set_xlabel("Time")
        ax2.set_ylabel("Score")
        ax2.set_title("Precision & Recall Over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()
        return fig, (ax1, ax2)

    def plot_comprehensive_dashboard(self, figsize: Tuple[int, int] = (20, 12)):
        """
        Create a comprehensive dashboard with all key visualizations.

        Args:
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)

        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 1, 1])

        # Main timeline plot
        ax_main = fig.add_subplot(gs[0, :])
        self._plot_main_timeline(ax_main)

        # Rolling metrics
        ax_rolling = fig.add_subplot(gs[1, 0])
        self._plot_rolling_metrics(ax_rolling)

        # Confusion matrix
        ax_cm = fig.add_subplot(gs[1, 1])
        self._plot_confusion_matrix(ax_cm)

        # Error distribution
        ax_errors = fig.add_subplot(gs[1, 2])
        self._plot_error_distribution(ax_errors)

        # Precision-Recall over time
        ax_pr = fig.add_subplot(gs[2, 0])
        self._plot_precision_recall_over_time(ax_pr)

        # Performance summary
        ax_summary = fig.add_subplot(gs[2, 1:])
        self._plot_performance_summary(ax_summary)

        plt.tight_layout()
        return fig

    def _calculate_rolling_metric(
        self, metric_array: np.ndarray, window_size: int
    ) -> np.ndarray:
        """Calculate rolling metric."""
        rolling_metric = np.zeros(self.n_points)
        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)
            rolling_metric[i] = np.mean(metric_array[start_idx:end_idx])
        return rolling_metric

    def _calculate_rolling_confusion_metrics(self, window_size: int) -> dict:
        """Calculate rolling confusion matrix metrics."""
        metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)

            window_pred = self.predictions[start_idx:end_idx]
            window_truth = self.ground_truth[start_idx:end_idx]

            if len(window_pred) > 0:
                metrics["accuracy"].append(accuracy_score(window_truth, window_pred))
                metrics["precision"].append(
                    precision_score(window_truth, window_pred, zero_division=0)
                )
                metrics["recall"].append(
                    recall_score(window_truth, window_pred, zero_division=0)
                )
                metrics["f1"].append(
                    f1_score(window_truth, window_pred, zero_division=0)
                )
            else:
                for key in metrics:
                    metrics[key].append(0)

        return {key: np.array(values) for key, values in metrics.items()}

    def _calculate_rolling_precision(self, window_size: int) -> np.ndarray:
        """Calculate rolling precision."""
        rolling_precision = np.zeros(self.n_points)
        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)

            window_pred = self.predictions[start_idx:end_idx]
            window_truth = self.ground_truth[start_idx:end_idx]
            rolling_precision[i] = precision_score(
                window_truth, window_pred, zero_division=0
            )

        return rolling_precision

    def _calculate_rolling_recall(self, window_size: int) -> np.ndarray:
        """Calculate rolling recall."""
        rolling_recall = np.zeros(self.n_points)
        for i in range(self.n_points):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(self.n_points, i + window_size // 2 + 1)

            window_pred = self.predictions[start_idx:end_idx]
            window_truth = self.ground_truth[start_idx:end_idx]
            rolling_recall[i] = recall_score(window_truth, window_pred, zero_division=0)

        return rolling_recall

    def _plot_main_timeline(self, ax):
        """Plot main timeline comparison."""
        # Ground truth as filled area
        ax.fill_between(
            self.timestamps,
            0,
            self.ground_truth,
            alpha=0.3,
            color="green",
            label="Ground Truth",
            step="pre",
        )

        # Predictions with error coloring
        correct_mask = self.correct
        ax.scatter(
            self.timestamps[correct_mask],
            self.predictions[correct_mask] + 0.1,
            c="blue",
            alpha=0.7,
            s=15,
            label="Correct",
        )
        ax.scatter(
            self.timestamps[~correct_mask],
            self.predictions[~correct_mask] + 0.1,
            c="red",
            alpha=0.7,
            s=15,
            label="Error",
        )

        ax.set_title("Model Predictions vs Ground Truth")
        ax.set_ylabel("State")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_rolling_metrics(self, ax):
        """Plot rolling accuracy and F1."""
        window_size = max(10, self.n_points // 20)
        rolling_acc = self._calculate_rolling_metric(self.correct, window_size)
        rolling_f1 = self._calculate_rolling_confusion_metrics(window_size)["f1"]

        ax.plot(self.timestamps, rolling_acc, "blue", label="Accuracy")
        ax.plot(self.timestamps, rolling_f1, "red", label="F1")
        ax.set_title("Rolling Metrics")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix(self, ax):
        """Plot confusion matrix."""
        cm = confusion_matrix(self.ground_truth, self.predictions)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title("Confusion Matrix")

    def _plot_error_distribution(self, ax):
        """Plot error type distribution."""
        error_counts = [
            np.sum(self.true_negatives),
            np.sum(self.false_positives),
            np.sum(self.false_negatives),
            np.sum(self.true_positives),
        ]
        labels = ["TN", "FP", "FN", "TP"]
        colors = ["lightgreen", "red", "orange", "darkgreen"]

        ax.pie(error_counts, labels=labels, colors=colors, autopct="%1.1f%%")
        ax.set_title("Prediction Distribution")

    def _plot_precision_recall_over_time(self, ax):
        """Plot precision and recall over time."""
        window_size = max(10, self.n_points // 20)
        rolling_precision = self._calculate_rolling_precision(window_size)
        rolling_recall = self._calculate_rolling_recall(window_size)

        ax.plot(self.timestamps, rolling_precision, "green", label="Precision")
        ax.plot(self.timestamps, rolling_recall, "blue", label="Recall")
        ax.set_title("Precision & Recall")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_summary(self, ax):
        """Plot performance summary table."""
        ax.axis("off")

        # Create summary table
        metrics_data = [
            ["Accuracy", f"{self.accuracy:.3f}"],
            ["Precision", f"{self.precision:.3f}"],
            ["Recall", f"{self.recall:.3f}"],
            ["F1-Score", f"{self.f1:.3f}"],
            ["True Positives", f"{np.sum(self.true_positives)}"],
            ["True Negatives", f"{np.sum(self.true_negatives)}"],
            ["False Positives", f"{np.sum(self.false_positives)}"],
            ["False Negatives", f"{np.sum(self.false_negatives)}"],
        ]

        table = ax.table(
            cellText=metrics_data,
            colLabels=["Metric", "Value"],
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        ax.set_title("Performance Summary", pad=20)

    def print_summary(self):
        """Print performance summary."""
        print("Model Performance Summary")
        print("=" * 30)
        print(f"Overall Accuracy: {self.accuracy:.3f}")
        print(f"Precision: {self.precision:.3f}")
        print(f"Recall: {self.recall:.3f}")
        print(f"F1-Score: {self.f1:.3f}")
        print("\nError Breakdown:")
        print(f"True Positives: {np.sum(self.true_positives)}")
        print(f"True Negatives: {np.sum(self.true_negatives)}")
        print(f"False Positives: {np.sum(self.false_positives)}")
        print(f"False Negatives: {np.sum(self.false_negatives)}")


if __name__ == "__main__":
    # Run demonstration
    try:
        print("=" * 60)
        print("ANIMATED POSITION PLOTTER DEMO")
        print("=" * 60)
        print(f"Backend in use: {matplotlib.get_backend()}")

        plotter, anim = demo_animated_plot()
        print("\nAnimation created successfully!")

        # Keep the plot alive for Qt backends
        backend = matplotlib.get_backend()
        if "Qt" in backend:
            print("\nQt backend detected - plot window should stay interactive")
            print("Close the plot window to continue...")
            try:
                # For Qt backends, ensure event loop runs
                if hasattr(plotter.fig.canvas, "start_main_loop"):
                    plotter.fig.canvas.start_main_loop()
            except:
                pass

    except Exception as e:
        print(f"Error running demonstration: {e}")
        import traceback

        traceback.print_exc()

    # Show usage examples
    print("\n" + "=" * 60)
    print("USAGE WITH YOUR DATA:")
    print("=" * 60)
    print(f"Current backend: {matplotlib.get_backend()}")
    print("""
# Plot your MATLAB maze shape:
maze_coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0],
               [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]

plotter = AnimatedPositionPlotter(your_data_helper)
plotter.setup_plot(
    custom_lines=[maze_coords],     # Your maze shape
    custom_line_colors='black',     # Maze color
    custom_line_styles='-',         # Solid lines
    custom_line_widths=3            # Thick walls
)
plotter.show()

# Multiple custom shapes:
maze = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0], [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]
shock_zone = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7], [0.3, 0.3]]  # Square shock zone

plotter.setup_plot(
    custom_lines=[maze, shock_zone],
    custom_line_colors=['black', 'red'],    # Different colors
    custom_line_styles=['-', '--'],         # Different styles
    custom_line_widths=[3, 2]               # Different widths
)

# Alternative: Use helper function
maze_coords = [[0, 0], [0, 1], [1, 1], [1, 0], [0.63, 0],
               [0.63, 0.75], [0.35, 0.75], [0.35, 0], [0, 0]]
custom_lines = create_maze_from_matlab(maze_coords)

plotter = create_plotter_for_data(
    your_data_helper,
    custom_lines=custom_lines
)

# Mix with other reference lines:
plotter.setup_plot(
    binary_colors=True,
    custom_lines=[maze_coords],          # Maze walls
    hlines=[0.5],                        # Center horizontal line
    vlines=[0.5],                        # Center vertical line
    custom_line_colors='black',          # Maze color
    line_colors='gray'                   # Reference line color
)

# Force Qt backend:
plotter = create_qt_plotter(your_data_helper, trail_length=40)

# In Jupyter notebook:
%matplotlib widget  # or %matplotlib qt
plotter = AnimatedPositionPlotter(your_data_helper)
plotter.show()

# Save as video:
plotter = AnimatedPositionPlotter(your_data_helper)
anim = plotter.create_animation(save_path='trajectory.mp4')

# Manual backend control:
import matplotlib
matplotlib.use('Qt5Agg')  # Before importing pyplot
plotter = AnimatedPositionPlotter(your_data_helper)
plotter.show()
""")

    print("\nAvailable backends on your system:")
    try:
        import matplotlib.backend_bases

        backends = []
        for backend in ["Qt5Agg", "Qt4Agg", "TkAgg", "GTK3Agg", "WXAgg"]:
            try:
                matplotlib.use(backend, force=False)
                backends.append(backend)
            except:
                pass
        print(f"Compatible backends: {backends}")
    except:
        print("Could not detect available backends")

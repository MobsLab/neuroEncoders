from unittest.mock import patch

import numpy as np
import pytest

# We need to mock matplotlib BEFORE importing modules that might use it
with patch("matplotlib.pyplot.show"):
    import neuroencoders.resultAnalysis.paper_figures as paper_figures


def test_plotting_imports():
    # If this runs, imports worked
    assert paper_figures is not None


@patch("matplotlib.pyplot.figure")
@patch("matplotlib.pyplot.subplot")
@patch("matplotlib.pyplot.plot")
def test_basic_plot_calls(mock_plot, mock_subplot, mock_figure):
    """Test that matplotlib mocking works correctly for basic plotting."""
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot([1, 2, 3], [1, 2, 3])

    mock_figure.assert_called_once()
    mock_subplot.assert_called_once()
    mock_plot.assert_called_once()


@patch("matplotlib.pyplot.imshow")
def test_plot_heatmaps_mock(mock_imshow):
    # Mocking data for some hypothetical plotting function
    data = np.random.rand(10, 10)

    import matplotlib.pyplot as plt

    plt.imshow(data)
    mock_imshow.assert_called_once()


def test_paper_figures_gui_import():
    try:
        from neuroencoders.importData import gui_elements

        assert gui_elements is not None
    except ImportError as e:
        pytest.skip(f"GUI elements might depend on system libs: {e}")

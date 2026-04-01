import matplotlib.pyplot as plt
import numpy as np

from neuroencoders.resultAnalysis.paper_figures import PaperFigures
from neuroencoders.resultAnalysis.print_results import overview_fig


# Mocking necessary parts
class MockPaperFigures(PaperFigures):
    def __init__(self):
        pass


def test_single_error_matrix():
    pf = MockPaperFigures()
    true_pos = np.random.rand(100)
    pred_pos = true_pos + np.random.normal(0, 0.05, 100)
    pred_pos = np.clip(pred_pos, 0, 1)

    fig, ax = plt.subplots()
    pf._plot_single_error_matrix(true_pos, pred_pos, ax=ax)
    plt.close(fig)
    print("test_single_error_matrix passed")


def test_overview_fig():
    # Test if print_results handles seconds
    pos = np.random.rand(100, 1)
    inferring = np.random.rand(100, 1)
    selection = np.ones(100, dtype=bool)
    timeStepsPred = np.arange(100) / 10.0  # seconds

    # Just check if it runs without error with dimOutput=1
    overview_fig(
        pos,
        inferring,
        selection,
        dimOutput=1,
        timeStepsPred=timeStepsPred,
        save=False,
        show=False,
    )
    print("test_overview_fig passed")


if __name__ == "__main__":
    try:
        test_single_error_matrix()
        test_overview_fig()
        print("All verification tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()

import argparse

import ephyviewer
import neo
import numpy as np
from ephyviewer import (
    AnalogSignalFromNeoRawIOSource,
    MainViewer,
    SpectrogramViewer,
    TraceViewer,
    mkQApp,
)

argparser = argparse.ArgumentParser(description="View Neuroscope data")
argparser.add_argument("--filename", "-f", type=str, help="Path to Neuroscope data")
args = argparser.parse_args()
# you must first create a main Qt application (for event loop)
app = mkQApp()

# create a fake signals 1 channel at 10kHz
# this emulate a moving frequency


# Create the main window that can contain several viewers
win = MainViewer(debug=True, show_auto_scale=True)

neorawio = neo.rawio.NeuroScopeRawIO(filename=args.filename)

source = AnalogSignalFromNeoRawIOSource(neorawio=neorawio)

# create a viewer for signal with TraceViewer
view1 = TraceViewer(source=source, name="trace")
view1.params["scale_mode"] = "same_for_all"
view1.params["xsize"] = 5.0
view1.auto_scale()


# create a SpectrogramViewer on the same source
view2 = SpectrogramViewer(source=source, name="spectrogram")

view2.params["xsize"] = 5.0
view2.params["colormap"] = "inferno"
view2.params["scalogram", "binsize"] = 0.1
view2.params["scalogram", "scale"] = "dB"
view2.params["scalogram", "scaling"] = "spectrum"

# add them to mainwindow
win.add_view(view1)
win.add_view(view2)


# show main window and run Qapp
win.show()
app.exec()

import argparse

import ephyviewer
import spikeinterface.extractors as se
from ephyviewer import (
    MainViewer,
    TimeFreqViewer,
    TraceViewer,
    mkQApp,
)

argparser = argparse.ArgumentParser(description="View Neuroscope data")
argparser.add_argument("--filename", "-f", type=str, help="Path to Neuroscope data")
args = argparser.parse_args()
# you must first create a main Qt application (for event loop)
app = mkQApp()
# Create the main window that can contain several viewers
win = MainViewer(debug=True, show_auto_scale=True)

recording = se.extractor_classes.NeuroScopeRecordingExtractor(file_path=args.filename)

sig_source = ephyviewer.SpikeInterfaceRecordingSource(recording=recording)

# create a viewer for signal with TraceViewer
view1 = TraceViewer(source=sig_source, name="signals")
view1.params["scale_mode"] = "same_for_all"
view1.params["xsize"] = 5.0
view1.auto_scale()

# create a time freq viewer connected to the same source
view2 = TimeFreqViewer(source=sig_source, name="tfr")

view2.params["xsize"] = 5.0
view2.params["show_axis"] = True
view2.params["timefreq", "deltafreq"] = 1
view2.params["timefreq", "f_start"] = 90
view2.params["timefreq", "f_stop"] = 250
view2.params["timefreq", "f0"] = 1
view2.params["timefreq", "normalisation"] = 0
view2.auto_scale()

# add them to mainwindow
win.add_view(view1)
win.add_view(view2)
view1.auto_scale()
view2.auto_scale()


# show main window and run Qapp
win.show()
app.exec()

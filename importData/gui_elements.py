import matplotlib as mplt
import numpy as np

mplt.use("TkAgg")
from tkinter import Button, Entry, Label, Tk, Toplevel


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

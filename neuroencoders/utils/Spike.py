#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:42:00 2020

@author: quarantine-charenton
"""


def openstruc(struc, name, L):
    from os.path import join

    import h5py

    if type(struc) != h5py._hl.dataset.Dataset:
        if "start" in list(struc.keys()):
            for i in range(len(L)):
                if L[i].endswith(name):
                    L[i] = name + "_intset"
        else:
            for k in list(struc.keys()):
                if name in L:
                    L.remove(name)
                L.append(join(name, k))
                openstruc(struc[k], join(name, k), L)


def ref2str(ref, file):
    import numpy as np

    out = []
    for i in range(len(ref)):
        try:
            L = list(np.squeeze(file[ref[i]][:]))
            if type(L[0]) == int:
                S = ""
                for l in L:
                    S += chr(l)
                    out.append(S)
            else:
                out.append(L)
        except TypeError:
            pass
    return out


# %%


class SpikeData:
    def __init__(self, path, time_unit="us"):
        import os

        import numpy as np

        # import utils.neuroseries as nts
        import pynapple as nts
        from scipy.io import loadmat

        if not path.endswith(".mat"):
            path = os.path.join(path, "SpikeData.mat")

        try:
            spikes = loadmat(path, squeeze_me=True)

        except NotImplementedError:
            import h5py

            spikes = h5py.File(path, "r")
            keys = list(spikes.keys())
            keys.remove("#refs#")

            if "S" in keys:
                keys.remove("S")
                Stemp = spikes["S/C"][:]
                Nb_clusters = len(Stemp)
                S = []
                for i in range(Nb_clusters):
                    SpkT = 100 * np.array(
                        np.array(spikes[Stemp[i][0]]["t"]).tolist()[0]
                    )
                    S.append(nts.Tsd(SpkT, np.nan * SpkT))
            self.S = S
            self.Nb_clusters = Nb_clusters
            self.info = {}
            for k in keys:
                if type(spikes[k]) == h5py._hl.dataset.Dataset:
                    try:
                        if type(np.squeeze(spikes[k][:])[0]) == h5py.h5r.Reference:
                            self.info[k] = ref2str(np.squeeze(spikes[k][:]), spikes)
                        else:
                            self.info[k] = np.squeeze(spikes[k][:])
                    except ValueError or AttributeError:
                        pass
                else:
                    L = []
                    openstruc(spikes[k], k, L)
                    for l in L:
                        if l.endswith("_intset"):
                            l = l[0:-7]
                            start = np.squeeze(spikes[l]["start"][:])
                            stop = np.squeeze(spikes[l]["stop"][:])
                            self.info[l] = nts.IntervalSet(
                                start, stop, time_units=time_unit
                            )
                        elif type(np.squeeze(spikes[l][:])[0]) == h5py.h5r.Reference:
                            self.info[l] = ref2str(np.squeeze(spikes[l][:]), spikes)

                        else:
                            self.info[l] = np.squeeze(spikes[l][:])

    def get_spikes(self, idx=None):
        import numpy as np

        if type(idx) == np.ndarray:
            idx = list(idx)
        if idx is None:
            return self.S
        elif type(idx) == int:
            return self.S[idx]
        elif type(idx) == list:
            return [self.S[i] for i in idx]
        elif type(idx[0]) == np.bool_:
            return [self.S[i] for i in range(len(idx)) if idx[i]]

    def features(self):
        return list(self.info.keys())

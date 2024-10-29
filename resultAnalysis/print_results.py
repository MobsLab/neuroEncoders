# Get libs
import os
import stat
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd

EC = np.array([45, 39])  # range of x and y in cm


def print_results(
    dir: str,
    show=False,
    typeDec: str = "NN",
    euclidean: bool = False,
    results=[],
    lossSelection: float = 0.3,
    windowSizeMS: int = 36,
):
    """
    This function is used to print the results of the decoding.
    args:
    - dir: the directory where the results are stored
    - show: if True, the figures will be shown
    - typeDec: the type of decoder used (NN or bayes)
    - euclidean: if True, the euclidean distance will be used
    - results: the results of the Decoding (mantadory to provide if typeDec is bayes)
    - lossSelection: the percentage of the best windows to selected
    - windowSizeMS: the size of the window in ms
    """
    outdir = os.path.join(dir, str(windowSizeMS))
    # Manage arguments
    if typeDec == "bayes" and not results:
        raise ValueError("You should provide results from BayesTrainer.test() function")

    block = show
    maxPos = 1
    # Get data
    if typeDec == "NN":
        predLossName = "predLoss"
        pos = pd.read_csv(
            os.path.expanduser(os.path.join(dir, str(windowSizeMS), "featureTrue.csv"))
        ).values[:, 1:]
        inferring = pd.read_csv(
            os.path.expanduser(os.path.join(dir, str(windowSizeMS), "featurePred.csv"))
        ).values[:, 1:]
        qControl = np.squeeze(
            pd.read_csv(
                os.path.expanduser(os.path.join(dir, str(windowSizeMS), "lossPred.csv"))
            ).values[:, 1:]
        )
        if os.path.isfile(
            os.path.expanduser(os.path.join(dir, str(windowSizeMS), "linearPred.csv"))
        ):
            linear = True
            lpos = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(dir, str(windowSizeMS), "linearTrue.csv")
                    )
                ).values[:, 1:]
            )
            linferring = np.squeeze(
                pd.read_csv(
                    os.path.expanduser(
                        os.path.join(dir, str(windowSizeMS), "linearPred.csv")
                    )
                ).values[:, 1:]
            )
        else:
            linear = False
    elif typeDec == "bayes":
        predLossName = "proba"
        pos = results["featureTrue"]
        inferring = results["featurePred"]
        qControl = np.squeeze(results[predLossName])
        if "linearPred" in results.keys():
            linear = True
            lpos = results["linearTrue"]
            linferring = results["linearPred"]
        else:
            linear = False
    else:
        raise ValueError('typeDec should be either "NN" or bayes"')
    dimOutput = pos.shape[1]
    assert pos.shape[1] == inferring.shape[1]
    if euclidean:
        pos = pos * EC
        inferring = inferring * EC

    # Save the executable
    if typeDec == "NN":
        with open(os.path.join(outdir, "reDrawFigures"), "w") as f:
            f.write(sys.executable + " " + os.path.abspath(__file__) + " " + dir)
        st = os.stat(os.path.join(outdir, "reDrawFigures"))
        os.chmod(os.path.join(outdir, "reDrawFigures"), st.st_mode | stat.S_IEXEC)

    # Get the best <loss_selection*100>% of data
    temp = qControl.argsort(axis=0)
    if typeDec == "NN":
        thresh = np.squeeze(qControl[temp[int(len(temp) * lossSelection)]])
        selection = np.squeeze(qControl < thresh)
    elif typeDec == "bayes":
        thresh = qControl[temp[int(len(temp) * (1 - lossSelection))]]
        selection = np.squeeze(qControl > thresh)
    frames = np.where(selection)[0]
    print(
        "total windows:",
        len(temp),
        "| selected windows:",
        len(frames),
        "(thresh",
        thresh,
        ")",
    )
    # Calculate 1d and 2d errors
    error = np.array(
        [np.linalg.norm(inferring[i, :] - pos[i, :]) for i in range(inferring.shape[0])]
    )  # eurledian distance
    print(
        "mean eucl. error:",
        np.nanmean(error) * maxPos,
        "| selected error:",
        np.nanmean(error[frames]) * maxPos,
    )
    if linear:
        lError = np.array(
            [np.abs((linferring[n] - lpos[n])) for n in range(linferring.shape[0])]
        )
        print(
            "mean linear error:",
            np.nanmean(lError),
            "| selected error:",
            np.nanmean(lError[frames]) * maxPos,
        )
    sys.stdout.write("threshold value: " + str(thresh) + "\r")
    sys.stdout.flush()
    # Plot things
    overview_fig(
        pos,
        inferring,
        selection,
        outdir,
        dimOutput=dimOutput,
        show=block,
        typeDec=typeDec,
    )
    fig_interror(
        pos,
        error,
        qControl,
        selection,
        thresh,
        outdir,
        dimOutput=dimOutput,
        show=block,
        typeDec=typeDec,
    )
    if linear:
        overview_fig(
            lpos,
            linferring,
            selection,
            outdir,
            dimOutput=1,
            show=block,
            typeDec=typeDec,
        )
        fig_interror(
            lpos,
            lError,
            qControl,
            selection,
            thresh,
            outdir,
            dimOutput=1,
            show=block,
            typeDec=typeDec,
        )

    ### Figures
    # Overview


def overview_fig(
    pos, inferring, selection, outfolder, dimOutput=2, show=False, typeDec="NN"
):
    if dimOutput == 2:
        fig, ax = plt.subplots(figsize=(15, 9))
        for dim in range(dimOutput):
            if dim > 0:
                ax1 = plt.subplot2grid((dimOutput, 1), (dim, 0), sharex=ax1)
            else:
                ax1 = plt.subplot2grid((dimOutput, 1), (dim, 0))
            ax1.plot(
                np.where(selection)[0],
                inferring[selection, dim],
                label="guessed dim" + str(dim) + " selection",
            )
            ax1.plot(pos[:, dim], label="true dim" + str(dim), color="xkcd:dark pink")
            ax1.legend()
            ax1.set_title("position " + str(dim))
    elif dimOutput == 1:
        _, ax2 = plt.subplots(figsize=(15, 9))
        ax2.plot(
            np.where(selection)[0],
            inferring[selection],
            label="guessed linear selection",
        )
        ax2.plot(pos, label="true linear", color="xkcd:dark pink")
        ax2.legend()
        ax2.set_title("linear position")
    plt.savefig(
        os.path.expanduser(
            os.path.join(outfolder, f"overviewFig_{dimOutput}d_{typeDec}.png")
        ),
        bbox_inches="tight",
    )
    if show:
        plt.show(block=show)
    print()

    # Interactive figure 2D


def fig_interror(
    pos,
    Error,
    q_control,
    selection,
    thresh,
    outfolder,
    dimOutput=2,
    show=False,
    typeDec="NN",
):
    fig, ax = plt.subplots(figsize=(15, 9))
    from matplotlib.widgets import Slider
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig.tight_layout(pad=3.0)
    ax = plt.subplot2grid((1, 2), (0, 0))
    if dimOutput == 2:
        s = plt.scatter(
            pos[selection, 0], pos[selection, 1], c=Error[selection], s=10
        )  # X and Y
        plt.axis("scaled")
    elif dimOutput == 1:
        s = plt.scatter(
            np.where(selection)[0], pos[selection], c=Error[selection], s=10
        )
    rangey = ax.get_ylim()[1] - ax.get_ylim()[0]
    rangex = ax.get_xlim()[1] - ax.get_xlim()[0]
    if dimOutput == 2:
        crange = max(rangex, rangey)
    elif dimOutput == 1:
        crange = rangey
    plt.clim(0, crange)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax)
    cbar.set_label("decoding error", rotation=270)
    ax.set_title("decoding error depending of mouse position")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    def update(val):
        sys.stdout.write("threshold value: " + str(val) + "\r")
        sys.stdout.flush()
        l.set_ydata([val, val])
        if typeDec == "NN":
            selection = np.squeeze(q_control < val)
        elif typeDec == "bayes":
            selection = np.squeeze(q_control > val)
        if dimOutput == 2:
            s.set_offsets(pos[selection, :])
            s.set_array(Error[selection])
            sel2 = selection[selNoNans]
            scatt.set_array(
                np.array([1 if sel2[n] else 0.2 for n in range(len(selNoNans))])
            )
        elif dimOutput == 1:
            s.set_offsets(np.c_[np.where(selection)[0], pos[selection]])
            s.set_array(Error[selection])
            sel2 = selection[selNoNans]
            scatt.set_array(
                np.array([1 if sel2[n] else 0.2 for n in range(len(selNoNans))])
            )
        fig.canvas.draw_idle()

    axcolor = "lightgoldenrodyellow"
    axfreq = plt.axes([0.1, 0.15, 0.25, 0.03], facecolor=axcolor)
    slider = Slider(
        axfreq,
        "selection",
        np.nanmin(q_control[np.isfinite(q_control)]),
        np.nanmax(q_control[np.isfinite(q_control)]),
        valinit=thresh,
        valstep=(
            np.nanmax(q_control[np.isfinite(q_control)])
            - np.nanmin(q_control[np.isfinite(q_control)])
        )
        / 100,
    )
    slider.on_changed(update)
    oldAx = ax

    # ERROR & STD
    ax = plt.subplot2grid((1, 2), (0, 1))
    selNoNans = np.squeeze(~np.isnan(Error))
    sel2 = selection[selNoNans]
    z = [1 if sel2[n] else 0.2 for n in range(len(sel2))]
    scatt = plt.scatter(
        Error[selNoNans],
        q_control[selNoNans],
        c=z,
        s=10,
        cmap=plt.cm.get_cmap("Greys"),
        vmin=0,
        vmax=1,
    )
    # plt.scatter(Error[selNoNans], inferring[selNoNans,dim_output], c=z, s=10)
    nBins = 20
    _, edges = np.histogram(Error[selNoNans], nBins)
    histIdx = []
    for bin in range(nBins):
        temp = []
        for n in range(len(Error)):
            if not selNoNans[n]:
                continue
            if Error[n] <= edges[bin + 1] and Error[n] > edges[bin]:
                temp.append(n)
        histIdx.append(temp)
    err = np.array(
        [
            [
                np.median(q_control[histIdx[n]])
                - np.percentile(q_control[histIdx[n]], 30)
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ],
            [
                np.percentile(q_control[histIdx[n]], 70)
                - np.median(q_control[histIdx[n]])
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ],
        ]
    )
    plt.errorbar(
        [(edges[n + 1] + edges[n]) / 2 for n in range(nBins) if len(histIdx[n]) > 10],
        [
            np.median(q_control[histIdx[n]])
            for n in range(nBins)
            if len(histIdx[n]) > 10
        ],
        c="xkcd:cherry red",
        yerr=err,
        label=r"$median \pm 20 percentile$",
        linewidth=3,
    )
    x_new = np.linspace(
        np.min(
            [
                (edges[n + 1] + edges[n]) / 2
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ]
        ),
        np.max(
            [
                (edges[n + 1] + edges[n]) / 2
                for n in range(nBins)
                if len(histIdx[n]) > 10
            ]
        ),
        num=len(Error),
    )
    coefs = poly.polyfit(
        [(edges[n + 1] + edges[n]) / 2 for n in range(nBins) if len(histIdx[n]) > 10],
        [
            np.median(q_control[histIdx[n]])
            for n in range(nBins)
            if len(histIdx[n]) > 10
        ],
        2,
    )
    # coefs = poly.polyfit(Error[selNoNans], inferring[selNoNans,dim_output], 2)
    ffit = poly.polyval(x_new, coefs)
    plt.plot(x_new, ffit, "k", linewidth=3)
    l = plt.axhline(thresh, c="k")
    ax.set_ylabel("evaluated loss")
    ax.set_xlabel("decoding error")
    ax.set_title("decoding error vs. evaluated loss")
    plt.savefig(
        os.path.expanduser(
            os.path.join(outfolder, f"errorFig_{dimOutput}d_{typeDec}.png")
        ),
        bbox_inches="tight",
    )
    if show:
        plt.show(block=show)
    print()


# Re-do (or do the figures)
if __name__ == "__main__":
    print_results(sys.argv[1], show=True)

    # # Movie
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax1 = plt.subplot2grid((1,1),(0,0))
    # im2, = ax1.plot([pos[0,1]*maxPos],[pos[0,0]*maxPos],marker='o', markersize=15, color="red")
    # im2b, = ax1.plot([inferring[0,1]*maxPos],[inferring[0,0]*maxPos],marker='P', markersize=15, color="green")

    # im3 = ax1.plot([125,170,170,215,215,210,60,45,45,90,90], [35,70,110,210,225,250,250,225,210,110,35], color="red")
    # im4 = ax1.plot([125,125,115,90,90,115,125], [100,215,225,220,185,100,100], color="red")
    # n = 135; nn=2*n
    # im4 = ax1.plot([nn-125,nn-125,nn-115,nn-90,nn-90,nn-115,nn-125], [100,215,225,220,185,100,100], color="red")
    # ax1.set_title('Decoding using full stack decoder', size=25)
    # ax1.get_xaxis().set_visible(False)
    # ax1.get_yaxis().set_visible(False)

    # def updatefig(frame, *args):
    #     reduced_frame = frame % len(frames)
    #     selected_frame = frames[reduced_frame]
    #     im2.set_data([pos[selected_frame,1]*maxPos],[pos[selected_frame,0]*maxPos])
    #     im2b.set_data([inferring[selected_frame,1]*maxPos],[inferring[selected_frame,0]*maxPos])
    #     return im2,im2b

    # save_len = len(frames)
    # ani = animation.FuncAnimation(fig,updatefig,interval=250, save_count=save_len)
    # ani.save(os.path.expanduser(folder+'/_tempMovie.mp4'))
    # fig.show()

    # np.savez(os.path.expanduser(fileName), pos, speed, inferring, trainLosses)
    # print('Results saved at:', fileName)

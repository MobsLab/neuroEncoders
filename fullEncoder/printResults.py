import os
import sys
import datetime
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from scipy.stats import gaussian_kde




findFolder = lambda path: path if path[-1]=='/' or len(path)==1 else findFolder(path[:-1])
folder = findFolder(sys.argv[1])
file = folder + 'results/inferring.npz'

results = np.load(os.path.expanduser(file))
pos = results['pos']
inferring = results['inferring']
trainLosses = results['trainLosses']
block=True
lossSelection = .2
maxPos = 1#253.92
dim_output = pos.shape[1]
assert(pos.shape[1] == inferring.shape[1]-1)




if trainLosses!=[]:
    fig, ax = plt.subplots(figsize=(8,8))
    colors = ['tab:red','tab:blue']
    ln1 = plt.semilogy(trainLosses[:,0], label="position loss", color = colors[0]) 
    ax.tick_params(axis='y', labelcolor=colors[0])
    ax2 = ax.twinx()
    ln2 = plt.semilogy(trainLosses[:,1], label="error loss", color = colors[1]) 
    ax2.tick_params(axis='y', labelcolor=colors[1])
    lns = ln1+ln2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc="upper right")
    ax.set_xlabel("training step")
    fig.tight_layout()
    plt.savefig(os.path.expanduser(folder+'results/lossFig.png'), bbox_inches='tight')
    # plt.show(block=block)


# ERROR & STD
Error = np.array([np.sqrt(sum([(inferring[n,dim] - pos[n,dim])**2 for dim in range(dim_output)])) for n in range(inferring.shape[0])])
selNoNans = ~np.isnan(Error)
fig = plt.figure(figsize=(8,8))
xy = np.vstack([Error[selNoNans], inferring[selNoNans,dim_output]])
z = gaussian_kde(xy)(xy)
plt.scatter(Error[selNoNans], inferring[selNoNans,dim_output], c=z, s=10)
nBins = 20
_, edges = np.histogram(Error[selNoNans], nBins)
histIdx = []
for bin in range(nBins):
    temp=[]
    for n in range(len(Error)):
        if not selNoNans[n]:
            continue
        if Error[n]<=edges[bin+1] and Error[n]>edges[bin]:
            temp.append(n)
    histIdx.append(temp)
err=np.array([
    [np.median(inferring[histIdx[n],dim_output])-np.percentile(inferring[histIdx[n],dim_output],30) for n in range(nBins) if len(histIdx[n])>10],
    [np.percentile(inferring[histIdx[n],dim_output],70)-np.median(inferring[histIdx[n],dim_output]) for n in range(nBins) if len(histIdx[n])>10]])
plt.errorbar(
    [(edges[n+1]+edges[n])/2 for n in range(nBins) if len(histIdx[n])>10],
    [np.median(inferring[histIdx[n],dim_output]) for n in range(nBins) if len(histIdx[n])>10], c='xkcd:cherry red', 
    yerr = err, 
    label=r'$median \pm 20 percentile$',
    linewidth=3)
x_new = np.linspace(np.min([(edges[n+1]+edges[n])/2 for n in range(nBins) if len(histIdx[n])>10]), np.max([(edges[n+1]+edges[n])/2 for n in range(nBins) if len(histIdx[n])>10]), num=len(Error))
coefs = poly.polyfit(
    [(edges[n+1]+edges[n])/2 for n in range(nBins) if len(histIdx[n])>10], 
    [np.median(inferring[histIdx[n],dim_output]) for n in range(nBins) if len(histIdx[n])>10], 
    2)
# coefs = poly.polyfit(Error[selNoNans], inferring[selNoNans,dim_output], 2)
ffit = poly.polyval(x_new, coefs)
plt.plot(x_new, ffit, 'k', linewidth=3)
ax = fig.axes[0]
ax.set_ylabel('evaluated loss')
ax.set_xlabel('decoding error')
plt.savefig(os.path.expanduser(folder+'results/errorFig.png'), bbox_inches='tight')
# plt.show(block=block)




temp = inferring[:,dim_output]
temp2 = temp.argsort()
# thresh = np.max(ffit)/3
thresh = temp[temp2[int(len(temp2)*lossSelection)]]
selection = inferring[:,dim_output]<thresh
frames = np.where(selection)[0]
print("total windows:",len(temp2),"| selected windows:",len(frames), "(thresh",thresh,")")



print('mean error:', np.nanmean(Error)*maxPos, "| selected error:", np.nanmean(Error[frames])*maxPos)




# Overview
fig, ax = plt.subplots(figsize=(15,9))
for dim in range(dim_output):
    if dim > 0:
        ax1 = plt.subplot2grid((dim_output,1),(dim,0), sharex=ax1)
    else:
        ax1 = plt.subplot2grid((dim_output,1),(dim,0))
    # ax1.plot(inferring[:,dim], label='guessed '+str(dim))
    ax1.plot(np.where(selection)[0], inferring[selection,dim], label='guessed dim'+str(dim)+' selection')
    ax1.plot(pos[:,dim], label='true dim'+str(dim), color='xkcd:dark pink')
    ax1.legend()
    ax1.set_title('position '+str(dim))
# plt.text(0,0,fileName)
plt.savefig(os.path.expanduser(folder+'results/overviewFig.png'), bbox_inches='tight')
plt.show(block=block)



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

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
file = folder + '_resultsForRnn_temp.npz'
# file = folder + '_resultsForRnn_2019-11-19_aligned.npz'
# fileName = folder + '_resultsForRnn_' + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")

results = np.load(os.path.expanduser(file))
Y_test = results['pos']
speed = results['spd']
testOutput = results['testOutput']
trainLosses = results['trainLosses']
block=True
lossSelection = .2
maxPos = 253.92


# fig = plt.figure(figsize=(8,8))
# plt.plot(trainLosses[:,0]) 
# plt.plot(trainLosses[:,1]) 
# plt.show(block=block)


# ERROR & STD
xError = np.abs(testOutput[:,0] - Y_test[:,0])
yError = np.abs(testOutput[:,1] - Y_test[:,1])
Error = np.array([np.sqrt(xError[n]**2 + yError[n]**2) for n in range(len(xError))])
fig = plt.figure(figsize=(8,8))
xy = np.vstack([Error, testOutput[:,2]])
z = gaussian_kde(xy)(xy)
plt.scatter(Error, testOutput[:,2], c=z, s=10)
nBins = 20
_, edges = np.histogram(Error, nBins)
histIdx = []
for bin in range(nBins):
    temp=[]
    for n in range(len(Error)):
        if Error[n]<=edges[bin+1] and Error[n]>edges[bin]:
            temp.append(n)
    histIdx.append(temp)
err=np.array([
    [np.median(testOutput[histIdx[n],2])-np.percentile(testOutput[histIdx[n],2],30) for n in range(nBins)],
    [np.percentile(testOutput[histIdx[n],2],70)-np.median(testOutput[histIdx[n],2]) for n in range(nBins)]])
plt.errorbar(
    [(edges[n+1]+edges[n])/2 for n in range(nBins)],
    [np.median(testOutput[histIdx[n],2]) for n in range(nBins)], c='xkcd:cherry red', 
    yerr = err, 
    label=r'$median \pm 20 percentile$',
    linewidth=3)
x_new = np.linspace(np.min(Error), np.max(Error), num=len(Error))
coefs = poly.polyfit(Error, testOutput[:,2], 2)
ffit = poly.polyval(x_new, coefs)
plt.plot(x_new, ffit, 'k', linewidth=3)
ax = fig.axes[0]
ax.set_ylabel('evaluated loss')
ax.set_xlabel('decoding error')
plt.savefig(os.path.expanduser(folder+'_errorFig.png'), bbox_inches='tight')
plt.show(block=block)




print('mean error is:', np.nanmean(Error)*maxPos)
tri = np.argsort(testOutput[:,2])
Selected_errors = np.array([ 
        np.nanmean(Error[ tri[0:1*len(tri)//10] ]), 
        np.nanmean(Error[ tri[1*len(tri)//10:2*len(tri)//10] ]),
        np.nanmean(Error[ tri[2*len(tri)//10:3*len(tri)//10] ]),
        np.nanmean(Error[ tri[3*len(tri)//10:4*len(tri)//10] ]),
        np.nanmean(Error[ tri[4*len(tri)//10:5*len(tri)//10] ]),
        np.nanmean(Error[ tri[5*len(tri)//10:6*len(tri)//10] ]),
        np.nanmean(Error[ tri[6*len(tri)//10:7*len(tri)//10] ]),
        np.nanmean(Error[ tri[7*len(tri)//10:8*len(tri)//10] ]),
        np.nanmean(Error[ tri[8*len(tri)//10:9*len(tri)//10] ]),
        np.nanmean(Error[ tri[9*len(tri)//10:len(tri)]       ]) ]) * maxPos
print("----Selected errors----")
print(Selected_errors)


temp = testOutput[:,2]
temp2 = temp.argsort()
thresh = temp[temp2[int(len(temp2)*lossSelection)]]
selection = testOutput[:,2]<thresh
frames = np.where(selection)[0]





# Overview
fig, ax = plt.subplots(figsize=(15,9))
ax1 = plt.subplot2grid((2,1),(0,0))
# ax1.plot(testOutput[:,0], label='guessed X')
ax1.plot(np.where(selection)[0], testOutput[selection,0], label='guessed X selection')
ax1.plot(Y_test[:,0], label='true X', color='xkcd:dark pink')
ax1.legend()
ax1.set_title('position X')

ax2 = plt.subplot2grid((2,1),(1,0), sharex=ax1)
# ax2.plot(testOutput[:,1], label='guessed Y')
ax2.plot(np.where(selection)[0], testOutput[selection,1], label='guessed Y selection')
ax2.plot(Y_test[:,1], label='true Y', color='xkcd:dark pink')
ax2.legend()
ax2.set_title('position Y')

# plt.text(0,0,fileName)
plt.savefig(os.path.expanduser(folder+'_overviewFig.png'), bbox_inches='tight')
plt.show(block=block)


# # Movie
# fig, ax = plt.subplots(figsize=(10,10))
# ax1 = plt.subplot2grid((1,1),(0,0))
# im2, = ax1.plot([Y_test[0,1]*maxPos],[Y_test[0,0]*maxPos],marker='o', markersize=15, color="red")
# im2b, = ax1.plot([testOutput[0,1]*maxPos],[testOutput[0,0]*maxPos],marker='P', markersize=15, color="green")

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
#     im2.set_data([Y_test[selected_frame,1]*maxPos],[Y_test[selected_frame,0]*maxPos])
#     im2b.set_data([testOutput[selected_frame,1]*maxPos],[testOutput[selected_frame,0]*maxPos])
#     return im2,im2b

# save_len = len(frames)
# ani = animation.FuncAnimation(fig,updatefig,interval=250, save_count=save_len)
# ani.save(os.path.expanduser(folder+'/_tempMovie.mp4'))
# fig.show()





# np.savez(os.path.expanduser(fileName), Y_test, speed, testOutput, trainLosses)
# print('Results saved at:', fileName)
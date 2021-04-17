using Pkg
Pkg.activate(".")
using CSV
using MAT
using ProgressBars
using HDF5
using LightXML
using Mmap
using Distributions
using Interpolations
using DataFrames
using TFRecord

# A small julia script to extract very efficiently
# spikes times and align them with a nnBehavior matrix.
# The results is stored in a CSV folder.

# The script should be launched with the following arguments, in order:
#   the path to the xml file
#   the path to the dat file
#   the path to the nnBehavior.mat file
#   the path at which the output csv file will be written
# e.g:
# xmlPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/continuous.xml"
# datPath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/continuous.dat"
# behavePath = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/nnBehavior.mat"
# fileName = "/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/test_spikeJulia.csv"
#
xmlPath = ARGS[2]
datPath = ARGS[3]
behavePath = ARGS[4]
fileName = ARGS[5]
datasetName = ARGS[6]
BUFFERSIZE = parse(Int64,ARGS[7])


print(xmlPath)

function extract_spike(xmlPath,datPath,behavePath,fileName,datasetName,BUFFERSIZE)
    #Read the xml file:
    xdoc = parse_file(xmlPath)
    xroot = root(xdoc)
    acquiSystem =  xroot["acquisitionSystem"]
    nbits = parse(Float16,content(acquiSystem[1]["nBits"][1]))
    Nchannel = parse(Float16,content(acquiSystem[1]["nChannels"][1]))
    samplingRate = parse(Float16,content(acquiSystem[1]["samplingRate"][1]))
    spd = xroot["spikeDetection"]
    groupList = spd[1]["channelGroups"][1]["group"]
    pint =s-> parse(Int64,s)
	#We need to add 1 to all channels because they are indexed starting at 0!!
    list_channels = map(s->pint.(map(c->content(c),s["channels"][1]["channel"])).+1,groupList)

    file = open(datPath)
    isreadable(file)
    #memory-map the file:
    mmapFile = Mmap.mmap(file,Matrix{Int16},(Int(Nchannel),Int(filesize(file)/(2.0*Nchannel))))[:,:]
    mmapFile = transpose(mmapFile)
    print("subtracting the low pass (fc=350Hz) filtered voltage to itself ")
    channel_focus = vcat(list_channels...)
    #apply the low pass filter on the mmapFile:
    α = exp(-2π*350.0/samplingRate)
    β = 1-α
    state = zeros(size(channel_focus,1))
    for id in tqdm(1:1:size(mmapFile,1))
        state = α*state .+ β*mmapFile[id,channel_focus].*0.195
        mmapFile[id,channel_focus] = mmapFile[id,channel_focus] .- Int16.(floor.(state./0.195))
    end
    #Note: to reduce the memory footprint we decide to round the filtering result to the upper integer

    behaveMat = h5open(behavePath)
    position_time = behaveMat["behavior"]["position_time"][:,:]
    speed = behaveMat["behavior"]["speed"][:,:]
	positions = behaveMat["behavior"]["positions"][:,:]

    nodes = (position_time[:,1],)
    nodes_index = float.(1:1:size(position_time,1))
    itp = interpolate(nodes, nodes_index, (Gridded(Constant())))

    # we should begin at spikes which timing are above the first position measure:
    possibleTime_spike  = collect((1:1:size(mmapFile,1))./samplingRate)
    minTest = possibleTime_spike.-position_time[1]
    minTest[minTest.<0] .= Inf
    spikeMin = argmin(abs.(minTest))
    maxTest = position_time[end].-possibleTime_spike
    maxTest[maxTest.<0] .= Inf
    spikeMax = argmin(abs.(maxTest))
    nodeTest = (possibleTime_spike[spikeMin:spikeMax],)
    spikePosIndex = itp.(nodeTest)

    posindexOfSpikes  = zeros(Int64,size(mmapFile,1))
    posindexOfSpikes[1:spikeMin-1] .= -1
    posindexOfSpikes[spikeMax+1:end] .= -1
    posindexOfSpikes[spikeMin:spikeMax] .= spikePosIndex[1]

    nGroups = size(list_channels,1)

    print("applying spike thresholds")
    #thresholds are computed for each channel in each group....
    possileSpike = map(id->zeros(Int64,size(mmapFile,1)),1:1:size(list_channels,1))
    possibleSpike_sum = map(id->zeros(Int64,size(mmapFile,1)),1:1:size(list_channels,1))
    noPreviousSpike = map(id->zeros(Int64,size(mmapFile,1)),1:1:size(list_channels,1))
    #let us filter by a std computed over a buffersize:
    for idb in ProgressBar(1:BUFFERSIZE:size(mmapFile,1)-BUFFERSIZE)
    	thresholds = map(l->3*std(mmapFile[idb:idb+BUFFERSIZE,l],dims=1),list_channels)
    	map(id->noPreviousSpike[id][idb:idb+BUFFERSIZE] .= prod(mmapFile[idb:idb+BUFFERSIZE,list_channels[id]] .>= -thresholds[id], dims=2)[:,1],1:1:size(list_channels,1))
    	map(id->possileSpike[id][idb:idb+BUFFERSIZE] .= sum(mmapFile[idb:idb+BUFFERSIZE,list_channels[id]] .< -thresholds[id], dims=2)[:,1],1:1:size(list_channels,1))
    	map(id->possibleSpike_sum[id][idb+1:idb+BUFFERSIZE] .= noPreviousSpike[id][idb:idb+BUFFERSIZE-1].*possileSpike[id][idb+1:idb+BUFFERSIZE],1:1:size(list_channels,1))
    	#special case of first value of the buffer:
    	if idb>1
    		map(id->possibleSpike_sum[id][idb] =  noPreviousSpike[id][idb-1]*possileSpike[id][idb],1:1:size(list_channels,1))
    	end
    end

    #
    # psa1 =  map(id->possibleSpike_sum[id].>=1,1:1:nGroups)
    # #other way to do it
    # delta = map(id->reduce(hcat,map(delta->psa1[id][16-delta:end-delta],0:1:14)),1:1:nGroups)
    # delta_summed = map(id->sum(delta[id],dims=2),1:1:nGroups)
    # unique(delta_summed[1])
    # sum_15steps = map(id->map(idp->sum(psa1[id][idp-15:idp]),tqdm(16:1:size(possibleSpike_sum[id],1)-15)),1:1:nGroups)
    # known_spikes = map(id->sum_15steps[id].==2,1:1:nGroups)
    # nb_known_sp = sum.(known_spikes)

    # function findtime(position_time,lastBestTime,time)
    #     for i in 1:1:(size(position_time,1)-lastBestTime)
    #         if abs(position_time[lastBestTime+i-1]-time)<abs(position_time[lastBestTime+i]-time)
    #             return lastBestTime+i-1
    #         end
    #     end
    #     return size(position_time,1)
    # end

    spikesFound = []
    lastBestTime = zeros(Int64,nGroups) .+ 1
    for index in tqdm(16:1:(size(mmapFile,1)-17-15)) #
        time = float(index)/samplingRate
        for i in 1:1:nGroups
            if possibleSpike_sum[i][index] >0
                # lastBestTime[i] = findtime(position_time,lastBestTime[i],time)
                # posiiton_index = lastBestTime[i]
                posiiton_index = posindexOfSpikes[index]
                #time, group, spike_index, position_index
                append!(spikesFound,[[time,i,index,posiiton_index]])
                # forbid other spikes in other channels of the group for 15 time-steps.
                possibleSpike_sum[i][index:index+14] .= 0
            end
        end
        if size(spikesFound,1)>100000
            spikesFound = transpose(reduce(hcat,spikesFound))
            CSV.write(fileName,(t=spikesFound[:,1],group=float(spikesFound[:,2]),
                                spike_id=float(spikesFound[:,3]),pos_id=float(spikesFound[:,4])),append=true,
                                header=["t", "group", "spike_id","pos_id"])
            spikesFound = []
        end
    end
    #write last batch of spikes:
    spikesFound = transpose(reduce(hcat,spikesFound))
    CSV.write(fileName,(t=spikesFound[:,1],group=float(spikesFound[:,2]),
                        spike_id=float(spikesFound[:,3]),pos_id=float(spikesFound[:,4])),append=true,
                        header=["t", "group", "spike_id","pos_id"])
    spikesFound = []


    #saving as a tensorflow tf.rec:

    window_length = 0.036
    function stack_spikes(spIndex,g)
    	if size(spIndex,1) ==0
    		return zeros((0,size(list_channels[g],1),32))
    	else
    		catspike = cat(map(s->mmapFile[Int(s)-15:Int(s)+16,list_channels[g]],spIndex)...,dims=3)
			# Julia and python do not ravel a tensor similarly,
			# For Julia, it progressively increases index from left to right gathering: [A[1,1,1],A[2,1,1],A[1,2,1],A[2,2,1],...]
			# Whereas Python stars from the end.
			# So that python reshape our array we thus store it by raveling from [32timsStepVoltageMeasure,NChannel,NspikesIn36msWindow]
			# And it will be read as [NspikesIn36msWindow,NChannel,32timsStepVoltageMeasure] in python!
			# So we don't permutedims!
			#permutedims(catspike,[3,2,1])
    		return catspike
    	end
    end

    spikesFound = CSV.read(fileName,DataFrame)
    spikesFound = Array(spikesFound)
    inEpoch = spikesFound[:,4] .> -1
    spikesFound = spikesFound[inEpoch,:]
    maxpos = maximum(positions)
    #🔥 : the group will be indexed from 0 to nGroups-1 in python
    # so we need to do the same in Julia
	t0 = spikesFound[1,1] #time of first spike:
	startindex = 1
	lastindex = 2
	feats = []
	for window_start in tqdm(t0:window_length:spikesFound[end,1]+window_length)
		if spikesFound[startindex,1]>=window_start && spikesFound[startindex,1]<(window_start+window_length)
			#find the first index where the spike time is beyond window_start
			while spikesFound[lastindex,1]<(window_start+window_length) && lastindex<(size(spikesFound,1)-1)
				lastindex = lastindex+1
			end

			spikeIndex = [spikesFound[startindex:lastindex-1,3][spikesFound[startindex:lastindex-1,2].==g] for g in 1:1:nGroups]
			spikes = [stack_spikes(spikeIndex[g],g) for g in 1:1:nGroups]

			# save into tensorflow the spike window:
			feat = Dict(
				"pos_index" => [Int(spikesFound[lastindex-1,4])-1],
				"pos" => Float32.(positions[Int(spikesFound[lastindex-1,4]),:]/maxpos),
				"length" =>[lastindex-startindex],
				"groups" => Int.(spikesFound[startindex:(lastindex-1),2] .-1),
				"time" => [Float32.(mean(spikesFound[startindex:(lastindex-1), 1]))]
			)
			# Pierre: convert the spikes dic into a tf.train.Feature, used for the tensorflow protocol.
			# their is no reason to change the key name but still done here.
			for g in 1:1:nGroups
				feat[string("group",g-1)] =  Array{Float32}(vcat(spikes[g]...).*0.195)
			end

			# next round will start at last index:
			startindex = lastindex
			append!(feats,[feat])
		end
	end
	TFRecord.write(datasetName,feats)
end


extract_spike(xmlPath,datPath,behavePath,fileName,datasetName,BUFFERSIZE)

#Note that the spikes are sorted by time in the array
# it will therefore be simple to group them into a spike window of 36ms bin.

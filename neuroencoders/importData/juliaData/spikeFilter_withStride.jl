using Pkg
Pkg.activate(".")
using CSV
using MAT
# using ProgressBars
using HDF5
using LightXML
using Mmap
using Distributions
using Interpolations
using DataFrames
using TFRecord
using Distributed
using Dates
using Printf

# A small julia script to extract very efficiently
# spikes times and align them with a nnBehavior matrix.
# The results is stored in a CSV folder.

# The script should be launched with the following arguments, in order:
#   the path to the xml file
#   the path to the dat file
#   the path to the nnBehavior.mat file
#   the path at which the output csv file will be written
# 	the path to the dataset where we will write the awake spike
# 	the path to the dataset where we will write the sleeping spike
# e.g:
# xmlPath = "~/dataTest/Mouse-M1199-1304/continuous.xml"
# datPath = "~/dataTest/Mouse-M1199-1304/continuous.dat"
# behavePath = "~/dataTest/Mouse-M1199-1304/nnBehavior.mat"
# fileName = "~/dataTest/Mouse-M1199-1304/test_spikeJulia.csv"

# #For testing purposes
# xmlPath = "/media/DataMOBS209/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/continuous.xml"
# datPath = "/media/DataMOBS209/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/continuous.dat"
# behavePath = "/media/DataMOBS209/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/nnBehavior.mat"
# fileName = "/media/DataMOBS209/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/tmpspikeData_fromJulia.csv"
# thresholdsFileName = replace(fileName, "spikeData_fromJulia" => "thresholds_Julia")
# datasetName = "/media/DataMOBS209/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/dataset/dataset_stride100.tfrec"
# datasetNameSleep = "/media/DataMOBS209/home/mobs/Documents/PierreCode/dataTest/Mouse-M1199-1304/dataset/datasetSleep_stride100.tfrec"
# BUFFERSIZE = 72000
# WINDOWSIZE = 0.200
# WINDOWSTRIDE = 0.200

xmlPath = ARGS[2]
datPath = ARGS[3]
behavePath = ARGS[4]
fileName = ARGS[5]
thresholdsFileName = replace(fileName, "spikeData_fromJulia" => "thresholds_Julia")
datasetName = ARGS[6]
datasetNameSleep = ARGS[7]
BUFFERSIZE = parse(Int64, ARGS[8])
WINDOWSIZE = parse(Float32, ARGS[9])
WINDOWSTRIDE = parse(Float32, ARGS[10])


mutable struct ProgressBar
    total::Int
    current::Int
    start_time::Dates.DateTime
    width::Int
    last_update::Dates.DateTime
    update_interval::Dates.Millisecond

    function ProgressBar(total::Int; width::Int=50, update_interval_ms::Int=200)
        new(total, 0, Dates.now(), width, Dates.now(), Dates.Millisecond(update_interval_ms))
    end
end

function update!(p::ProgressBar, current::Int)
    p.current = current
    now_time = Dates.now()

    # Only update display if enough time has passed
    if now_time - p.last_update >= p.update_interval
        p.last_update = now_time
        display_progress(p)
    end
end

function display_progress(p::ProgressBar)
    elapsed = Dates.now() - p.start_time
    percent = p.current / p.total

    # Calculate remaining time
    if percent > 0
        total_seconds_estimated = Dates.value(elapsed) / 1000 / percent
        remaining_seconds = total_seconds_estimated - Dates.value(elapsed) / 1000

        # Format the time remaining
        hours, remainder = divrem(remaining_seconds, 3600)
        minutes, seconds = divrem(remainder, 60)
        time_str = @sprintf("%02d:%02d:%02d", hours, minutes, round(seconds))
    else
        time_str = "--:--:--"
    end

    # Create the progress bar
    filled = round(Int, percent * p.width)
    bar = string(
        "█"^filled,
        "░"^(p.width - filled),
    )

    # Print the progress bar
    print("\r")
    print(@sprintf(" %3d%% |%s| %d/%d [ETA: %s]",
        round(Int, percent * 100),
        bar,
        p.current,
        p.total,
        time_str
    ))

    # Flush output buffer to ensure display updates immediately
    flush(stdout)

    # Add a newline if we're done
    if p.current >= p.total
        println()
    end
end

using CodecZlib
using BufferedStreams
#we modify the TFrecord so it Effectively appends
function TFRecord.write(s::AbstractString, x; compression=nothing, bufsize=1024 * 1024)
    io = BufferedOutputStream(open(s, "a"), bufsize)
    if compression == :gzip
        io = GzipCompressorStream(io)
    elseif compression == :zlib
        io = ZlibCompressorStream(io)
    else
        isnothing(compression) || throw(ArgumentError("unsupported compression method: $compression"))
    end
    TFRecord.write(io, x)
    close(io)
end


function isInEpochs(time, epochs)
    # for an epochs array in the format [[start,end,start,end,....]]
    # verify if time is in any of these epochs:
    if size(epochs, 1) > 0
        return map(t -> sum((t .>= epochs[1:2:end-1, 1]) .* (t .< epochs[2:2:end, 1])) > 0, time)
    else
        return map(t -> false, time)
    end
end


function extract_spike_with_buffer(xmlPath, datPath, behavePath, fileName, datasetName, datasetNameSleep, BUFFERSIZE, WINDOWSIZE)
    #Read the xml file:
    xdoc = parse_file(xmlPath)
    xroot = root(xdoc)
    acquiSystem = xroot["acquisitionSystem"]
    nbits = parse(Float16, content(acquiSystem[1]["nBits"][1]))
    @assert nbits == 16
    Nchannel = parse(Float16, content(acquiSystem[1]["nChannels"][1]))
    samplingRate = parse(Float64, content(acquiSystem[1]["samplingRate"][1])) # increase sampling rate to avoid time drift issues
    spd = xroot["spikeDetection"]
    groupList = spd[1]["channelGroups"][1]["group"]
    pint = s -> parse(Int64, s)
    #We need to add 1 to all channels because they are indexed starting at 0!!
    list_channels = map(s -> pint.(map(c -> content(c), s["channels"][1]["channel"])) .+ 1, groupList)

    println("Number of bits: ", nbits)
    println("Number of channel: ", Nchannel)
    println("Sampling rate: ", samplingRate)
    println("List of channels for each group: ", list_channels)
    println("Window size: ", string(WINDOWSIZE * 1000), " ms")
    println("Window stride: ", string(WINDOWSTRIDE * 1000), " ms")

    file = open(datPath)
    isreadable(file)
    #memory-map the file (as it's Int16, one element is 2 bytes. Hence there 
    # are filesize/(2*Nchannel) elements in the matrix to mmap):
    mmapFile = Mmap.mmap(file, Matrix{Int16}, (Int(Nchannel), Int(filesize(file) / (2.0 * Nchannel))))
    #WARNING: I guess a non-Int16 acquisition system won't work there ?
    mmapFile = transpose(mmapFile)
    channel_focus = vcat(list_channels...)

    # first we will associate each spike to a measurement of environmental feature
    # and the time at which it was measured
    behaveMat = h5open(behavePath)
    position_time = behaveMat["behavior"]["position_time"][:, :]
    speed = behaveMat["behavior"]["speed"][:, :]
    positions = behaveMat["behavior"]["positions"][:, :]
    sleepPeriods = []
    try
        # sleepPeriods = vcat(behaveMat["behavior"]["sleepPeriods"][:]...)
        # sleepPeriods = reshape(sleepPeriods,(size(sleepPeriods,1),1))
        sleepPeriods = behaveMat["behavior"]["sleepPeriods"][:, :]
        sleepPeriods = reshape(sleepPeriods, (size(sleepPeriods, 2), 1))
        println("Detected sleep periods in nnBehavior.mat")
    catch
        println("No sleep periods detected in nnBehavior.mat, saving all spikes in the position_time limit in the same dataset")
        sleepPeriods = []
    end
    maxpos = maximum(positions)

    nodes = (position_time[:, 1],)
    nodes_index = float.(1:1:size(position_time, 1))
    println("Interpolating the position_time to get the position index of each spike")
    itp = interpolate(nodes, nodes_index, (Gridded(Constant())))

    nGroups = size(list_channels, 1)

    println("subtracting the low pass (fc=350Hz) filtered voltage to itself ")

    buffer_mmap = zeros(Float32, BUFFERSIZE + 15 + 16, size(mmapFile, 2))
    #15 at the beginning to complete early spikes of the buffer
    #16 at the end to complete final spike of the buffer
    # why these numbers? it would seem that behavior Data has a ~14.4Hz sampling frequency + 32 timepoints waveforms is what we want to capture
    state = zeros(Float32, size(channel_focus, 1))
    old_possibleSpike_sum = map(id -> zeros(Int64, 14), 1:1:nGroups)
    α = Float32(exp(-2π * 350.0 / samplingRate))
    β = Float32(1 - α)
    thresholds = [zeros(Float32, (1, size(l, 1))) for l in list_channels]

    feats = [] #list storing all spikes fromatted as Dict for tensorflow
    lastBuffIndex = (size(mmapFile, 1) % BUFFERSIZE == 0) ? div(size(mmapFile, 1), BUFFERSIZE) : div(size(mmapFile, 1), BUFFERSIZE) + 1

    progress = ProgressBar(lastBuffIndex)
    for idBuff in 1:lastBuffIndex
        # Théotime: this loop is where it usually fails.
        # Error: LoadError: BoundsError: attempt to access 7-element Array{Any,1} at index [8]
        # we should begin at spikes which timing are above the first position measure:
        possibleTime_spike = collect(float.(1+(idBuff-1)*BUFFERSIZE:1:min(size(mmapFile, 1), BUFFERSIZE * idBuff)) ./ samplingRate)
        # we need to be careful with the last buffer, which might contain less elements
        posindexOfSpikes = zeros(Int64, size(possibleTime_spike, 1)) #of size BUFFERSIZE or, for the last buffer, less
        if position_time[1, 1] > possibleTime_spike[end]
            # no position should be saved:
            posindexOfSpikes .= -1
        elseif position_time[end, 1] < possibleTime_spike[1]
            # no position should be saved:
            posindexOfSpikes .= -1
        elseif position_time[1, 1] >= possibleTime_spike[1]
            # the buffer spreads over the min limit of position
            minTest = possibleTime_spike .- position_time[1, 1]
            minTest[minTest.<0] .= Inf
            spikeMin = argmin(abs.(minTest))
            if position_time[end, 1] > possibleTime_spike[end]
                nodeTest = (possibleTime_spike[spikeMin:end],)
                spikePosIndex = itp.(nodeTest)
                posindexOfSpikes[1:spikeMin-1] .= -1
                posindexOfSpikes[spikeMin:end] .= spikePosIndex[1]
            else
                maxTest = position_time[end, 1] .- possibleTime_spike
                maxTest[maxTest.<0] .= Inf
                spikeMax = argmin(abs.(maxTest))
                nodeTest = (possibleTime_spike[spikeMin:spikeMax],)
                spikePosIndex = itp.(nodeTest)
                posindexOfSpikes[1:spikeMin-1] .= -1
                posindexOfSpikes[spikeMin:spikeMax] .= spikePosIndex[1]
                posindexOfSpikes[spikeMax+1:end] .= -1
            end
        elseif position_time[1, 1] < possibleTime_spike[1]
            if position_time[end, 1] <= possibleTime_spike[end]
                # the buffer spreads over the max limit of position
                maxTest = position_time[end, 1] .- possibleTime_spike
                maxTest[maxTest.<0] .= Inf
                spikeMax = argmin(abs.(maxTest))
                nodeTest = (possibleTime_spike[1:spikeMax],)
                spikePosIndex = itp.(nodeTest)
                posindexOfSpikes[1:spikeMax] .= spikePosIndex[1]
                posindexOfSpikes[spikeMax+1:end] .= -1
            else
                # the buffer is included inside the max and the min limit of position
                nodeTest = (possibleTime_spike[1:end],)
                spikePosIndex = itp.(nodeTest)
                posindexOfSpikes .= spikePosIndex[1]
            end
        end
        #finally, we assign to each time step that belongs to a sleep period a posindex of -2
        # as referenced by the sleepEpcohs in nnBehavior.mat
        # which consists of an array [start,end,start,end,....] of sleep epochs
        # Note: for future purpose (decoding of other env feature present in sleep)
        # or decoding of feature found in the neurons activity this might not be the best
        # way to go. Notably if we want to train during sleep!
        posindexOfSpikes[isInEpochs(possibleTime_spike, sleepPeriods)] .= -2

        # first buffer: 15 first time step are set at 0
        # other buffers: we already have filtered the end of the previous buffer (15 time steps) and beginning of the buffer (16 time steps)
        startid = idBuff == 1 ? 15 + 1 : 15 + 16 + 1
        #take care of last buffer:
        lastid = idBuff * BUFFERSIZE + 16 > size(mmapFile, 1) ? (size(mmapFile, 1) - (idBuff - 1) * BUFFERSIZE - 1 - 16) + startid : size(buffer_mmap, 1)
        buffer_mmap[startid:lastid, :] = idBuff == 1 ? mmapFile[1:min(BUFFERSIZE + 16, size(mmapFile, 1)), :] .* 0.195 : mmapFile[(idBuff-1)*BUFFERSIZE+16+1:min(idBuff * BUFFERSIZE + 16, size(mmapFile, 1)), :] .* 0.195
        #apply the low pass filter on the mmapFile:
        for id in (startid:1:size(buffer_mmap, 1))
            temp = buffer_mmap[id, channel_focus] .- state
            state = α * state .+ β * buffer_mmap[id, channel_focus]
            buffer_mmap[id, channel_focus] .= temp
        end
        #Note: to reduce the memory footprint we decide to round the filtering result to the upper integer
        # 16/6/2021: modified by adding a temporal variable as in Intan filter.
        #


        # print("applying spike thresholds")
        sidThresh = idBuff == 1 ? 1 : (15 + 1) #start index to look for new spikes
        #thresholds are computed for each channel in each group....
        possileSpike = map(id -> zeros(Int64, BUFFERSIZE), 1:1:size(list_channels, 1))
        possibleSpike_sum = map(id -> zeros(Int64, BUFFERSIZE), 1:1:size(list_channels, 1))
        noPreviousSpike = map(id -> zeros(Int64, BUFFERSIZE), 1:1:size(list_channels, 1))

        #note: changed to see the effect of a varying threshold:
        #if idBuff == 1
        #    for tppl in enumerate(list_channels)
        #        id,l = tppl
        #        thresholds[id] .= 3*std(buffer_mmap[sidThresh:end-16,l],dims=1)
        #    end
        #end
        thresholds = map(l -> 3 * std(buffer_mmap[sidThresh:end-16, l], dims=1), list_channels)

        dicThresholds = Dict()
        map(i -> dicThresholds[i] = thresholds[i], 1:1:size(thresholds, 1))
        CSV.write(thresholdsFileName, dicThresholds, append=true)

        map(id -> noPreviousSpike[id][1:end] .= prod(buffer_mmap[sidThresh:sidThresh+BUFFERSIZE-1, list_channels[id]] .>= -thresholds[id], dims=2)[:, 1], 1:1:size(list_channels, 1))
        map(id -> possileSpike[id][1:end] .= sum(buffer_mmap[sidThresh:sidThresh+BUFFERSIZE-1, list_channels[id]] .< -thresholds[id], dims=2)[:, 1], 1:1:size(list_channels, 1))
        map(id -> possibleSpike_sum[id][2:end] .= noPreviousSpike[id][1:end-1] .* possileSpike[id][2:end], 1:1:size(list_channels, 1))
        #note: still work for the last buffer: the buffer_mmap is initialized at 0, so unless thresholds are at 0, which will never be the case
        #	we obtain possileSpike =0 for the time step beyond the end of the last buffer, and will therefore have no influence
        # to the rest of the algo.
        #special case of first value of the buffer:
        if idBuff > 1
            map(id -> possibleSpike_sum[id][1] = possileSpike[id][sidThresh], 1:1:size(list_channels, 1))
            #so we have to save noPreviousSpike and garbage collect it correctly too.
        else
            #for the first buffer we don't have previous voltage measure so we accept to use the spike as is.
            map(id -> possibleSpike_sum[id][1] = possileSpike[id][1], 1:1:size(list_channels, 1))
        end

        #finally we need to check if there was a spike in the 14 last bin of the previous possibleSpike_sum buffer...
        for index_before in 1:1:14
            for i in 1:1:nGroups
                if old_possibleSpike_sum[i][index_before] > 0
                    possibleSpike_sum[i][1:index_before] .= 0
                end
            end
        end

        spikesFound = []
        for index in 1:1:BUFFERSIZE #
            const_dt = 1.0 / samplingRate # precise float64 to avoid precision issue
            time = float(index + (idBuff - 1) * BUFFERSIZE) * const_dt
            for i in 1:1:nGroups
                if possibleSpike_sum[i][index] > 0 && (index + (idBuff - 1) * BUFFERSIZE <= size(mmapFile, 1) - 16) #forbid too late spike because we would miss the info to capture waveform
                    # lastBestTime[i] = findtime(position_time,lastBestTime[i],time)
                    # posiiton_index = lastBestTime[i]
                    posiiton_index = posindexOfSpikes[index]
                    #time, group, spike_index, position_index
                    append!(spikesFound, [[time, i, index + (idBuff - 1) * BUFFERSIZE, posiiton_index]])
                    # forbid other spikes in other channels of the group for 14 time-steps.
                    possibleSpike_sum[i][index:min(index + 14, size(possibleSpike_sum[i], 1))] .= 0
                end
            end
        end
        spikesBehaveFound = []
        spikesSleepFound = []
        #write last batch of spikes:
        if size(spikesFound, 1) > 0
            spikesFound = transpose(reduce(hcat, spikesFound))
            CSV.write(fileName, (t=spikesFound[:, 1], group=float(spikesFound[:, 2]),
                    spike_id=float(spikesFound[:, 3]), pos_id=float(spikesFound[:, 4])), append=true,
                header=["t", "group", "spike_id", "pos_id"])
            #indicate that the last spikes were saved and can be used to forbid early spikes in the next buffer
            map(id -> old_possibleSpike_sum[id] .= possibleSpike_sum[id][end-13:end], 1:1:nGroups)
            #saving as a tensorflow tf.rec:

            window_length = WINDOWSIZE
            function stack_spikes(spIndex, g)
                #spIndex is the index of the spike in the buffer
                if size(spIndex, 1) == 0
                    #check if there is any spike in the buffer
                    return zeros((0, size(list_channels[g], 1), 32))
                else
                    spIndex = spIndex .- (idBuff - 1) * BUFFERSIZE .+ 15 #correct to set at the right position in the buffer
                    catspike = cat(map(s -> buffer_mmap[Int(s)-15:Int(s)+16, list_channels[g]], spIndex)..., dims=3)
                    # Julia and python do not ravel a tensor similarly,
                    # For Julia, it progressively increases index from left to right gathering: [A[1,1,1],A[2,1,1],A[1,2,1],A[2,2,1],...]
                    # Whereas Python starts from the end.
                    # So that python reshape our array we thus store it by raveling from [32timsStepVoltageMeasure,NChannel,NspikesIn36msWindow]
                    # And it will be read as [NspikesIn36msWindow,NChannel,32timsStepVoltageMeasure] in python!
                    # So we don't permutedims!
                    #permutedims(catspike,[3,2,1])
                    return catspike
                end
            end
            inBehaveEpoch = spikesFound[:, 4] .> -1
            spikesBehaveFound = spikesFound[inBehaveEpoch, :]
            inSleepEpoch = spikesFound[:, 4] .== -2
            spikesSleepFound = spikesFound[inSleepEpoch, :]
        end

        function saveSpikesFound(spikesFound)
            feats = []
            if size(spikesFound, 1) > 0 #if there is any spike to extract:
                #🔥 : the group will be indexed from 0 to nGroups-1 in python
                # so we need to do the same in Julia

                #Let us find the start of windows aligned on the first spike of windows of length window_stride (36ms)
                currentFirstSpikeTime = spikesFound[1, 1]
                currentFirstSpikeId = 1
                startWind = []
                for i in 1:1:size(spikesFound, 1)
                    if (spikesFound[i, 1] > currentFirstSpikeTime + WINDOWSTRIDE) # we find the beginning of the next window
                        append!(startWind, [currentFirstSpikeId])
                        currentFirstSpikeId = i
                        currentFirstSpikeTime = spikesFound[i, 1]
                    else
                        if (i == size(spikesFound, 1))
                            append!(startWind, [currentFirstSpikeId])
                            #    println("unfinished 36 ms window at buffer limit")
                        end
                    end
                end

                #For each of these start we find the stopindex =(last spike id)+1 which is less than window_length from the starting spike
                stopWind = []
                currentWindow = 1
                lastSpikeId = size(spikesFound, 1)
                for i in 1:1:size(spikesFound, 1)
                    if (spikesFound[i, 1] > spikesFound[startWind[currentWindow], 1] + window_length) # we find the beginning of the next window
                        append!(stopWind, [i])
                        currentWindow = currentWindow + 1
                    end
                end
                # Account for situations when WINDOWSIZE > WINDOWSTRIDE (late windows in the buffer are shorter)
                while (size(stopWind, 1) != size(startWind, 1))
                    append!(stopWind, [lastSpikeId + 1])
                    currentWindow = currentWindow + 1
                end
                # println("unfinished longer window at buffer limit")

                for i in 1:1:size(stopWind, 1)
                    startindex = startWind[i]
                    lastindex = stopWind[i]

                    spikeIndex = [spikesFound[startindex:lastindex-1, 3][spikesFound[startindex:lastindex-1, 2].==g] for g in 1:1:nGroups]
                    spikes = [stack_spikes(spikeIndex[g], g) for g in 1:1:nGroups]

                    # save into tensorflow the spike window:
                    pos_id_last = Int(spikesFound[lastindex-1, 4])
                    if pos_id_last == -2
                        feat = Dict(
                            "pos_index" => [-2],
                            "pos" => Float32.([0, 0]),
                            "length" => [lastindex - startindex],
                            "groups" => Int.(spikesFound[startindex:(lastindex-1), 2] .- 1),
                            "time" => [Float32.(mean(spikesFound[startindex:(lastindex-1), 1]))], # the mean time of the spikes in the window
                            "time_behavior" => Float32.([position_time[clamp(1, 1, size(position_time, 1)), 1]]),  # time of the first position measure from nnBehavior.mat
                            "indexInDat" => Int.(spikesFound[startindex:(lastindex-1), 3] .- 1)
                        )
                    else
                        feat = Dict(
                            "pos_index" => [Int(spikesFound[lastindex-1, 4]) - 1],
                            "pos" => Float32.(positions[Int(spikesFound[lastindex-1, 4]), :] / maxpos),
                            "length" => [lastindex - startindex],
                            "groups" => Int.(spikesFound[startindex:(lastindex-1), 2] .- 1),
                            "time" => [Float32.(mean(spikesFound[startindex:(lastindex-1), 1]))],
                            "time_behavior" => Float32.([position_time[pos_id_last, 1]]),  # behavior time of the last spike in the window
                            "indexInDat" => Int.(spikesFound[startindex:(lastindex-1), 3] .- 1)
                        )
                    end
                    # Pierre: convert the spikes dic into a tf.train.Feature, used for the tensorflow protocol.
                    # their is no reason to change the key name but still done here.
                    for g in 1:1:nGroups
                        feat[string("group", g - 1)] = Array{Float32}(vcat(spikes[g]...) .* 0.195)
                    end
                    append!(feats, [feat])
                end
            end
            return feats
        end
        feats = saveSpikesFound(spikesBehaveFound)
        if size(feats, 1) > 0
            TFRecord.write(datasetName, feats)
        end
        feats = saveSpikesFound(spikesSleepFound)
        if size(feats, 1) > 0
            TFRecord.write(datasetNameSleep, feats)
        end

        buffer_end = buffer_mmap[end-15-16+1:end]
        #save the end of the buffer mmap as we will use it again
        #without filtering it once more.
        buffer_mmap = zeros(Float32, 1) #before: Int16
        feats = []
        GC.gc()  #garbage collect the buffer as well as the feat dictionaries and all other arrays
        buffer_mmap = zeros(Float32, BUFFERSIZE + 15 + 16, size(mmapFile, 2))
        buffer_mmap[1:31] .= buffer_end

        # Update progress bar
        update!(progress, idBuff)
    end
end

extract_spike_with_buffer(xmlPath, datPath, behavePath, fileName, datasetName, datasetNameSleep, BUFFERSIZE, WINDOWSIZE)

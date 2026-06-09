using Pkg
Pkg.activate(".")
using CSV
using MAT
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
using Base.Threads
using LinearAlgebra
using CodecZlib
using BufferedStreams

# Read parameters from command line arguments
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

    if now_time - p.last_update >= p.update_interval
        p.last_update = now_time
        display_progress(p)
    end
end

function display_progress(p::ProgressBar)
    elapsed = Dates.now() - p.start_time
    percent = p.current / p.total

    if percent > 0
        total_seconds_estimated = Dates.value(elapsed) / 1000 / percent
        remaining_seconds = total_seconds_estimated - Dates.value(elapsed) / 1000

        hours, remainder = divrem(remaining_seconds, 3600)
        minutes, seconds = divrem(remainder, 60)
        time_str = @sprintf("%02d:%02d:%02d", hours, minutes, round(seconds))
    else
        time_str = "--:--:--"
    end

    filled = round(Int, percent * p.width)
    bar = string("█"^filled, "░"^(p.width - filled))

    print("\r")
    print(@sprintf(" %3d%% |%s| %d/%d [ETA: %s]", round(Int, percent * 100), bar, p.current, p.total, time_str))
    flush(stdout)

    if p.current >= p.total
        println()
    end
end

# Override TFRecord.write for direct file appending
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
    if size(epochs, 1) > 0
        return map(t -> sum((t .>= epochs[1:2:end-1, 1]) .* (t .< epochs[2:2:end, 1])) > 0, time)
    else
        return map(t -> false, time)
    end
end

struct SpikeRecord
    time::Float64
    group::Int64
    spike_id::Int64
    pos_id::Int64
end

function extract_spike_parallel_safe(xmlPath, datPath, behavePath, fileName, thresholdsFileName, datasetName, datasetNameSleep, BUFFERSIZE, WINDOWSIZE, WINDOWSTRIDE)
    # 1. Parse XML Configuration
    xdoc = parse_file(xmlPath)
    xroot = root(xdoc)
    acquiSystem = xroot["acquisitionSystem"]
    Nchannel = parse(Int64, content(acquiSystem[1]["nChannels"][1]))
    samplingRate = parse(Float64, content(acquiSystem[1]["samplingRate"][1]))
    const_dt = 1.0 / samplingRate # precise float64 to avoid precision issue
    spd = xroot["spikeDetection"]
    groupList = spd[1]["channelGroups"][1]["group"]

    pint = s -> parse(Int64, s)
    list_channels = [pint.(map(c -> content(c), g["channels"][1]["channel"])) .+ 1 for g in groupList]
    channel_focus = vcat(list_channels...)
    nGroups = length(list_channels)

    # 2. Open Files & Memory Map
    file = open(datPath)
    mmapFile = Mmap.mmap(file, Matrix{Int16}, (Int(Nchannel), Int(filesize(file) ÷ (2 * Nchannel))))
    mmapFile_T = transpose(mmapFile)
    total_samples = size(mmapFile_T, 1)

    behaveMat = h5open(behavePath)
    position_time = behaveMat["behavior"]["position_time"][:, :]
    positions = behaveMat["behavior"]["positions"][:, :]
    sleepPeriods = haskey(behaveMat["behavior"], "sleepPeriods") ? behaveMat["behavior"]["sleepPeriods"][:, :] : Matrix{Float64}(undef, 0, 0)
    if !isempty(sleepPeriods)
        sleepPeriods = reshape(sleepPeriods, (size(sleepPeriods, 2), 1))
    end
    maxpos = maximum(positions)

    # Interpolator
    nodes = (position_time[:, 1],)
    nodes_index = float.(1:size(position_time, 1))
    itp = interpolate(nodes, nodes_index, (Gridded(Constant())))

    α = Float32(exp(-2π * 350.0 / samplingRate))
    β = Float32(1.0 - α)

    lastBuffIndex = ceil(Int, total_samples / BUFFERSIZE)
    WARMUP_SAMPLES = 2000

    all_chunks_spikes = Vector{Vector{SpikeRecord}}(undef, lastBuffIndex)
    all_chunks_thresholds = Vector{Dict{String,Any}}(undef, lastBuffIndex)

    progress = ProgressBar(lastBuffIndex)
    progress_lock = ReentrantLock()

    println("Processing $(lastBuffIndex) buffers across $(Threads.nthreads()) threads...")

    println("Will save thresholds file to $(thresholdsFileName)")

    # --- PHASE 1: HYPER-FAST MATHEMATICS LOOP (No Locks, Max CPU Scaling) ---
    Threads.@threads for idBuff in 1:lastBuffIndex
        sample_start = (idBuff - 1) * BUFFERSIZE + 1
        sample_end = min(total_samples, idBuff * BUFFERSIZE)
        current_chunk_len = sample_end - sample_start + 1

        actual_start = max(1, sample_start - WARMUP_SAMPLES)
        warmup_len = sample_start - actual_start

        local_buffer = zeros(Float32, current_chunk_len + warmup_len, Nchannel)
        local_state = zeros(Float32, length(channel_focus))

        @views local_buffer[1:(current_chunk_len+warmup_len), :] .= mmapFile_T[actual_start:sample_end, :] .* 0.195f0

        # Run IIR Filter Warm-up
        @inbounds for id in 1:warmup_len
            for (ch_idx, ch) in enumerate(channel_focus)
                local_state[ch_idx] = α * local_state[ch_idx] + β * local_buffer[id, ch]
            end
        end

        # Run Real Filtering Window
        @inbounds for id in (warmup_len+1):(warmup_len+current_chunk_len)
            for (ch_idx, ch) in enumerate(channel_focus)
                temp = local_buffer[id, ch] - local_state[ch_idx]
                local_state[ch_idx] = α * local_state[ch_idx] + β * local_buffer[id, ch]
                local_buffer[id, ch] = temp
            end
        end

        sidThresh = warmup_len + 1
        end_idx = warmup_len + current_chunk_len

        spikesFound = SpikeRecord[]
        possibleTime_spike = collect((sample_start:sample_end) ./ samplingRate)
        posindexOfSpikes = fill(-1, length(possibleTime_spike))

        for idx in eachindex(possibleTime_spike)
            t = possibleTime_spike[idx]
            if t >= position_time[1, 1] && t <= position_time[end, 1]
                posindexOfSpikes[idx] = Int(itp(t))
            end
        end
        if !isempty(sleepPeriods)
            posindexOfSpikes[isInEpochs(possibleTime_spike, sleepPeriods)] .= -2
        end

        possileSpike = [zeros(Int64, current_chunk_len) for _ in 1:nGroups]
        possibleSpike_sum = [zeros(Int64, current_chunk_len) for _ in 1:nGroups]
        noPreviousSpike = [zeros(Int64, current_chunk_len) for _ in 1:nGroups]
        old_possibleSpike_sum = [zeros(Int64, 14) for _ in 1:nGroups]

        if idBuff > 1
            boundary_start = max(1, sample_start - 14)
            boundary_len = sample_start - boundary_start
            for g in 1:nGroups
                ch_group = list_channels[g]
                @views thresh_vals_b = 3.0f0 .* std(local_buffer[sidThresh:end_idx, ch_group], dims=1)
                @views b_buff = local_buffer[(sidThresh-boundary_len):(sidThresh-1), ch_group]

                for b_idx in 1:boundary_len
                    if any(b_buff[b_idx, :] .< -thresh_vals_b)
                        old_possibleSpike_sum[g][15-boundary_len+b_idx-1] = 1
                    end
                end
            end
        end

        local_thresholds_to_save = Dict{String,Any}()
        for g in 1:nGroups
            ch_group = list_channels[g]
            @views thresh_vals = 3.0f0 .* std(local_buffer[sidThresh:end_idx, ch_group], dims=1)
            local_thresholds_to_save["group_$(g-1)"] = thresh_vals

            @views begin
                noPreviousSpike[g][1:current_chunk_len] .= prod(local_buffer[sidThresh:(sidThresh+current_chunk_len-1), ch_group] .>= -thresh_vals, dims=2)[:, 1]
                possileSpike[g][1:current_chunk_len] .= sum(local_buffer[sidThresh:(sidThresh+current_chunk_len-1), ch_group] .< -thresh_vals, dims=2)[:, 1]
            end

            possibleSpike_sum[g][2:current_chunk_len] .= noPreviousSpike[g][1:current_chunk_len-1] .* possileSpike[g][2:current_chunk_len]
            possibleSpike_sum[g][1] = possileSpike[g][1]

            for index_before in 1:14
                if old_possibleSpike_sum[g][index_before] > 0
                    possibleSpike_sum[g][1:index_before] .= 0
                end
            end
        end

        for index in 1:current_chunk_len
            global_idx = index + (idBuff - 1) * BUFFERSIZE
            if global_idx > total_samples - 16
                break
            end

            if index > current_chunk_len - 16 && global_idx + 16 <= total_samples
                continue
            end

            time = float(global_idx) * const_dt
            for g in 1:nGroups
                if possibleSpike_sum[g][index] > 0
                    push!(spikesFound, SpikeRecord(time, g, global_idx, posindexOfSpikes[index]))
                    stop_range = min(index + 14, current_chunk_len)
                    possibleSpike_sum[g][index:stop_range] .= 0
                end
            end
        end
        all_chunks_spikes[idBuff] = spikesFound
        all_chunks_thresholds[idBuff] = local_thresholds_to_save

        lock(progress_lock) do
            update!(progress, idBuff)
        end
    end

    # --- PHASE 2: STREAM TO DISK SEQUENTIALLY (Fast Continuous NVMe Writing) ---
    println("All buffers calculated. Writing to disk storage arrays...")

    for idBuff in 1:lastBuffIndex
        spikesFound = all_chunks_spikes[idBuff]
        local_thresholds_to_save = all_chunks_thresholds[idBuff]

        if !isempty(local_thresholds_to_save)
            CSV.write(thresholdsFileName, local_thresholds_to_save, append=true)
        end

        if !isempty(spikesFound)
            csv_t = [s.time for s in spikesFound]
            csv_group = [Float64(s.group) for s in spikesFound]
            csv_spike_id = [Float64(s.spike_id) for s in spikesFound]
            csv_pos_id = [Float64(s.pos_id) for s in spikesFound]

            CSV.write(fileName, (t=csv_t, group=csv_group, spike_id=csv_spike_id, pos_id=csv_pos_id),
                append=true, header=["t", "group", "spike_id", "pos_id"])

            spikesBehaveFound = [s for s in spikesFound if s.pos_id > -1]
            spikesSleepFound = [s for s in spikesFound if s.pos_id == -2]

            # Reconstruct the original scaling window parameters cleanly
            window_length = WINDOWSIZE

            # Reconstruct the waveform snippets safely from the persistent global memory map file
            function stack_spikes_parallel(local_spikes, g_idx)
                if isempty(local_spikes)
                    return zeros(Float32, 0, length(list_channels[g_idx]), 32)
                end

                matrices = map(local_spikes) do s
                    # Compute absolute indices into continuous file coordinates directly
                    g_start = max(1, s.spike_id - 15)
                    g_end = min(total_samples, s.spike_id + 16)

                    # Read directly from memory map file view, bypassing localized thread scope limits
                    @views raw_snippet = Matrix{Float32}(mmapFile_T[g_start:g_end, list_channels[g_idx]]) .* 0.195f0

                    # Emulate running the IIR filter dynamically on the target snippet view to match expectations
                    snippet_state = zeros(Float32, length(list_channels[g_idx]))
                    @inbounds for r_idx in 1:size(raw_snippet, 1)
                        for c_idx in 1:size(raw_snippet, 2)
                            temp = raw_snippet[r_idx, c_idx] - snippet_state[c_idx]
                            snippet_state[c_idx] = α * snippet_state[c_idx] + β * raw_snippet[r_idx, c_idx]
                            raw_snippet[r_idx, c_idx] = temp
                        end
                    end
                    return raw_snippet
                end
                return cat(matrices..., dims=3)
            end

            function saveSpikesFound_parallel(target_spikes)
                feats = Dict{String,Any}[]
                if isempty(target_spikes)
                    return feats
                end

                currentFirstSpikeTime = target_spikes[1].time
                currentFirstSpikeId = 1
                startWind = Int[]

                for i in eachindex(target_spikes)
                    if target_spikes[i].time > currentFirstSpikeTime + WINDOWSTRIDE
                        push!(startWind, currentFirstSpikeId)
                        currentFirstSpikeId = i
                        currentFirstSpikeTime = target_spikes[i].time
                    elseif i == length(target_spikes)
                        push!(startWind, currentFirstSpikeId)
                    end
                end

                stopWind = Int[]
                currentWindow = 1
                lastSpikeId = length(target_spikes)
                for i in eachindex(target_spikes)
                    if target_spikes[i].time > target_spikes[startWind[currentWindow]].time + window_length
                        push!(stopWind, i)
                        currentWindow += 1
                    end
                end
                while length(stopWind) != length(startWind)
                    push!(stopWind, lastSpikeId + 1)
                end

                for i in eachindex(stopWind)
                    startindex = startWind[i]
                    lastindex = stopWind[i]
                    window_sub = @view target_spikes[startindex:(lastindex-1)]

                    spikes = [stack_spikes_parallel([s for s in window_sub if s.group == g], g) for g in 1:nGroups]

                    pos_id_last = Int(target_spikes[lastindex-1].pos_id)
                    mean_time = mean([s.time for s in window_sub])
                    idx_dat = [Int(s.spike_id - 1) for s in window_sub]
                    grp_ids = [Int(s.group - 1) for s in window_sub]

                    if pos_id_last == -2
                        feat = Dict(
                            "pos_index" => [-2],
                            "pos" => Float32[0.0, 0.0],
                            "length" => [lastindex - startindex],
                            "groups" => grp_ids,
                            "time" => [Float32(mean_time)],
                            "time_behavior" => Float32[position_time[1, 1]],
                            "indexInDat" => idx_dat
                        )
                    else
                        feat = Dict(
                            "pos_index" => [pos_id_last - 1],
                            "pos" => Float32.(positions[pos_id_last, :] / maxpos),
                            "length" => [lastindex - startindex],
                            "groups" => grp_ids,
                            "time" => [Float32(mean_time)],
                            "time_behavior" => Float32[position_time[pos_id_last, 1]],
                            "indexInDat" => idx_dat
                        )
                    end

                    for g in 1:nGroups
                        feat["group$(g - 1)"] = Array{Float32}(vcat(spikes[g]...) .* 0.195f0)
                    end
                    push!(feats, feat)
                end
                return feats
            end

            behave_feats = saveSpikesFound_parallel(spikesBehaveFound)
            if !isempty(behave_feats)
                TFRecord.write(datasetName, behave_feats)
            end

            sleep_feats = saveSpikesFound_parallel(spikesSleepFound)
            if !isempty(sleep_feats)
                TFRecord.write(datasetNameSleep, sleep_feats)
            end
        end
    end

    close(file)
    close(behaveMat)
end

extract_spike_parallel_safe(xmlPath, datPath, behavePath, fileName, thresholdsFileName, datasetName, datasetNameSleep, BUFFERSIZE, WINDOWSIZE, WINDOWSTRIDE)

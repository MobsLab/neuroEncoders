using MAT

path = "./../../../dataTest/LisaRouxDataset/PositionProbes.mat"
probPos = matread(path)
path = "./../../../dataTest/LisaRouxDataset/TaskPosition.mat"
taskPos = matread(path)

path = "./../../../dataTest/LisaRouxDataset/RestPeriods.mat"
restPeriod = matread(path)
rest = hcat([restPeriod["PreRestPeriod"],restPeriod["PostRestPeriod"]]...)[1,:]

positions = vcat([probPos["PreProbePosition"][4:end,2:3],taskPos["Position"][4:end,2:3],probPos["PostProbePosition"][4:end,2:3]]...)
positions_times = vcat([probPos["PreProbePosition"][4:end,1],taskPos["Position"][4:end,1],probPos["PostProbePosition"][4:end,1]]...)
speedpre = sqrt.((sum(probPos["PreProbePosition"][5:end,2:3] .- probPos["PreProbePosition"][4:end-1,2:3],dims=2)).^2)
speedpre = append!([speedpre[1]],speedpre)
speedtask = sqrt.((sum(taskPos["Position"][5:end,2:3] .- taskPos["Position"][4:end-1,2:3],dims=2)).^2)
speedtask = append!([speedtask[1]],speedtask)
speedpost = sqrt.((sum(probPos["PostProbePosition"][5:end,2:3] .- probPos["PostProbePosition"][4:end-1,2:3],dims=2)).^2)
speedpost = append!([speedpost[1]],speedpost)
speeds  = vcat([speedpre,speedtask,speedpost]...)

path = "./../../../dataTest/LisaRouxDataset/ProbePeriods.mat"
probePeriod = matread(path)
path = "./../../../dataTest/LisaRouxDataset/TaskPeriod.mat"
taskPeriod = matread(path)


dictsave = Dict(
  "behavior" => Dict(
      "positions" => positions,
      "position_time"=>reshape(positions_times,(size(positions_times,1),1)),
      "speed" => reshape(speeds,(size(speeds,1),1)),
      "sleepPeriods" => reshape(rest,(size(rest,1),1)),
      "SessionNames" => reshape(["PreProbe","PostProbe","task"],(1,3)),
      "SessionStart" => reshape([probePeriod["PreProbePeriod"][1,1],probePeriod["PostProbePeriod"][1,1],taskPeriod["TaskPeriod"][1,1]],(1,3)),
      "SessionStop" => reshape([probePeriod["PreProbePeriod"][1,2],probePeriod["PostProbePeriod"][1,2],taskPeriod["TaskPeriod"][1,2]],(1,3)))
)
pathout = "./../../../dataTest/LisaRouxDataset/nnBehavior.mat"
matwrite(pathout,dictsave)
matread(pathout)

# file = open("./../../../dataTest/LisaRouxDataset/M007_S07_07222015.dat")
# msize = filesize(file)
# l = Int(msize/(2*75))
# mmapFile = Mmap.mmap(file,Matrix{Int16},(Int(75),Int(filesize(file)/(2.0*75))))
# mmapFile = transpose(mmapFile)

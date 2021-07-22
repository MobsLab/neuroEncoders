function extractTsd=extractTsd(folderData)
% the function should receive the folder ending by a /
    %%%%%%%%%%%--- load the tsd file to read tsds array ---%%%%%%%%%%%
    addpath('./tsdPackage/')

	%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
	Behavior=importdata(strcat(folderData,'behavResources.mat'));
	disp('Data Loaded.')
    
    X = Data(Behavior.("AlignedXtsd"));
    Y = Data(Behavior.("AlignedYtsd"));
    V = Data(Behavior.("Vtsd"));
    behavior.positions     = [X Y];
    behavior.position_time = Range(Behavior.("AlignedXtsd"))/10000;
    behavior.speed = V;

    if isfield(Behavior,'SessionEpoch')
        sessionNames = fieldnames(Behavior.SessionEpoch)
        behavior.SessionNames =  {}
        behavior.SessionStart = []
        behavior.SessionStop = []
        for k = 1:length(sessionNames)
            epochk = Behavior.SessionEpoch.(sessionNames{k})
            behavior.SessionStart(k) = [Start(epochk)/10000]
            behavior.SessionStop(k)= [Stop(epochk)/10000]
            behavior.SessionNames(k) = {sessionNames{k}}
        end
        behavior.sleepPeriods = []
    else
        behavior.SessionNames = {"Recording"}
        behavior.Start = [Start(behavior.position_time)]
        behavior.Stop = [Stop(behavior.position_time)]
        behavior.sleepPeriods = []
    end
    save(strcat(folderData,'nnBehavior.mat'),'behavior','-v7.3');
end
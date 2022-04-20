function extractTsd(folderData, target)
% the function should receive the folder ending by a /
    %%%%%%%%%%%--- load the tsd file to read tsds array ---%%%%%%%%%%%
    addpath('./tsdPackage/')
    
    folderData = [folderData filesep];
	%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
	Behavior=importdata(strcat(folderData,'behavResources.mat'));
	disp('Data Loaded.')
    
    disp(['target: ', target]);
	if strcmp(target, 'pos')
        try
            X = Data(Behavior.("AlignedXtsd"));
            Y = Data(Behavior.("AlignedYtsd"));
            T = Range(Behavior.("AlignedXtsd"), 's');
        catch
            X = Data(Behavior.("Xtsd"));
            Y = Data(Behavior.("Ytsd"));
            T = Range(Behavior.("Xtsd"), 's');
        end
        V = Data(Behavior.("Vtsd"));
		behavior.positions     = [X Y];
		behavior.position_time = T;
        behavior.speed = V;
	else
		behavior.positions     = Data(Behavior.(target));
		behavior.position_time = Range(Behavior.(target), 's');
    end
    
    if isfield(Behavior,'SessionEpoch')
        sessionNames = fieldnames(Behavior.SessionEpoch);
        behavior.SessionNames =  {};
        behavior.SessionStart = [];
        behavior.SessionStop = [];
        for k = 1:length(sessionNames)
            epochk = Behavior.SessionEpoch.(sessionNames{k});
            behavior.SessionStart(k) = Start(epochk, 's');
            behavior.SessionStop(k)= Stop(epochk, 's');
            behavior.SessionNames(k) = {sessionNames{k}};
        end
        behavior.sleepPeriods = [];
    else
        behavior.SessionNames =  {};
        behavior.SessionNames(1) = {'Recording'};
        behavior.SessionStart = [];
        behavior.SessionStart(1) = behavior.position_time(1);
        behavior.SessionStop = [];
        behavior.SessionStop(1) =behavior.position_time(end);
        behavior.sleepPeriods = [];
    end
    
    
    %% Get sleep periods if thy exist
    if isfield(Behavior,'SessionEpoch')
        sleep_pattern = arrayfun(@(x) strfind(lower(behavior.SessionNames{x}), 'sleep'), 1:length(behavior.SessionNames), ...
            'UniformOutput', false); % Find it with no register
        bool_sleep = arrayfun(@(x) ~isempty(sleep_pattern{x}), 1:length(sleep_pattern));
        id_sleep = find(bool_sleep);
        if sum(bool_sleep) > 0
            behavior.sessionSleepNames = behavior.SessionNames(bool_sleep);
            
            behavior.sleepPeriods = [behavior.SessionStart(id_sleep(1)) behavior.SessionStop(id_sleep(1))];
            if sum(bool_sleep) > 1
                for isleep = 2:length(id_sleep)
                    behavior.sleepPeriods = [behavior.sleepPeriods ...
                        behavior.SessionStart(id_sleep(isleep)) behavior.SessionStop(id_sleep(isleep))];
                end
            end
        end
    end
    
    save(strcat(folderData,'nnBehavior.mat'),'behavior','-v7.3');
    disp('Behavior is successfully extracted')
end
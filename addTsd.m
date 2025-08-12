function addTsd(folderData, target)
% the function should receive the folder ending by a /
%%%%%%%%%%%--- load the tsd file to read tsds array ---%%%%%%%%%%%

% documentation: this function is used to extract additional behavior data from the
% tsd file and add it to the nnBehavior.mat file
% input: folderData: the folder where the tsd file is located
%        target: the name of the tsd to extract
% output: the behavior data is appended in a mat file which contains the
%         following fields:
%         - behavior: the behavior data already extracted from the tsd file
%         - optional: a new struct with the following TSDs added:
%           - FreezeEpoch: the epochs where the animal is freezing
%           - StimEpoch: the epochs where the animal is freezing
%           - tRipples: the epochs where the animal has ripples
%


addpath('./tsdPackage/')

folderData = [folderData filesep];
%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%
Behavior=importdata(strcat(folderData,'behavResources.mat'));
optional = struct();
disp('Data Loaded.')
optional.start_freeze = Start(Behavior.("FreezeEpoch"),'s');
optional.stop_freeze = Stop(Behavior.("FreezeEpoch"), 's');
optional.start_stim = Start(Behavior.("StimEpoch"), 's');
optional.stop_stim = Stop(Behavior.("StimEpoch"), 's');
optional.PosMat = Behavior.("PosMat");

try
    SWR=importdata(strcat(folderData,'SWR.mat'));

    optional.tRipples = Range(SWR.tRipples, 's');
catch
    disp('No SWR data found, skipping tRipples extraction.')
end

% Save the behavior data
save(strcat(folderData,'nnBehavior.mat'),'optional','-append', '-v7.3');
save(strcat(folderData,'optional_nnBehavior.mat'),'optional', '-v7.3');
disp('Behavior is successfully extracted')
end


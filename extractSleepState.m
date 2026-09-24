function extractTsd = extractTsd(folderData)
% EXTRACTTSD Extracts sleep state epochs and saves them to nnSleepScoring.mat
% The function expects folderData to be a string/char path to the data folder.

% Ensure tsdPackage path is correctly added relative to this script
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'tsdPackage'));

disp("Will change directory to: " + folderData)

% Safely change directory and ensure we return to the original on exit/error
originalDir = pwd;
cleanUp = onCleanup(@() cd(originalDir));
cd(folderData);

%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%

% Try loading OBGamma first, fallback to Accelero if missing
if exist('SleepScoring_OBGamma.mat', 'file') == 2
    sleepScore = load('SleepScoring_OBGamma.mat');
elseif exist('SleepScoring_Accelero.mat', 'file') == 2
    sleepScore = load('SleepScoring_Accelero.mat');
else
    error('Neither SleepScoring_OBGamma.mat nor SleepScoring_Accelero.mat was found.');
end
disp('Data Loaded successfully.')

%%%%%%%%%%%--- EXTRACT STATES ---%%%%%%%%%%%

% REM Epochs
rem = struct('remStart', [], 'remStop', []);
if isfield(sleepScore, 'REMEpoch')
    rem.remStart = Start(sleepScore.REMEpoch) / 10000;
    rem.remStop = Stop(sleepScore.REMEpoch) / 10000;
end

% SWS Epochs
sws = struct('swsStart', [], 'swsStop', []);
if isfield(sleepScore, 'SWSEpoch')
    sws.swsStart = Start(sleepScore.SWSEpoch) / 10000;
    sws.swsStop = Stop(sleepScore.SWSEpoch) / 10000;
end

% Wake Epochs (Optional fallback/addition)
wake = struct('wakeStart', [], 'wakeStop', []);
if isfield(sleepScore, 'WakeEpoch')
    wake.wakeStart = Start(sleepScore.WakeEpoch) / 10000;
    wake.wakeStop = Stop(sleepScore.WakeEpoch) / 10000;
elseif isfield(sleepScore, 'Wake')
    wake.wakeStart = Start(sleepScore.Wake) / 10000;
    wake.wakeStop = Stop(sleepScore.Wake) / 10000;
end

% Micro-wake Epochs
microwake = struct('microwakeStart', [], 'microwakeStop', []);
if isfield(sleepScore, 'microWakeEpochOB')
    microwake.microwakeStart = Start(sleepScore.microWakeEpochOB) / 10000;
    microwake.microwakeStop = Stop(sleepScore.microWakeEpochOB) / 10000;
elseif isfield(sleepScore, 'microWakeEpochAcc')
    microwake.microwakeStart = Start(sleepScore.microWakeEpochAcc) / 10000;
    microwake.microwakeStop = Stop(sleepScore.microWakeEpochAcc) / 10000;
end


% Noise Epochs
noise = struct('noiseStart', [], 'noiseStop', []);
if isfield(sleepScore, 'TotalNoiseEpoch')
    noise.noiseStart = Start(sleepScore.TotalNoiseEpoch) / 10000;
    noise.noiseStop = Stop(sleepScore.TotalNoiseEpoch) / 10000;
end

%%%%%%%%%%%--- SAVE OUTPUT ---%%%%%%%%%%%

% Save structural data with v7.3 compatibility for Python/mat73 reading
save('nnSleepScoring.mat', 'rem', 'sws', 'wake', 'microwake', 'noise', '-v7.3');
disp('nnSleepScoring.mat generated.');

% Return structural array as output if requested
extractTsd.rem = rem;
extractTsd.sws = sws;
extractTsd.wake = wake;
extractTsd.microwake = microwake;
extractTsd.noise = noise;
end

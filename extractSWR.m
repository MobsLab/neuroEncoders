function extractTsd=extractTsd(folderData)
% the function should receive the folder ending by a /
%%%%%%%%%%%--- load the tsd file to read tsds array ---%%%%%%%%%%%
scriptDir = fileparts(mfilename('fullpath'));
addpath(fullfile(scriptDir, 'tsdPackage'));
cd(folderData)
%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%

try
    SWR = importdata('SWR.mat');
catch
    try
        SWR = importdata('ripples.mat');
    catch
        SWR = importdata('Ripples.mat');
    end
end
disp('Data Loaded.')

ripple = SWR.("ripples");

save('nnSWR.mat','ripple','-v7.3');
end

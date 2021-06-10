function extractTsd=extractTsd(folderData)
% the function should receive the folder ending by a /
    %%%%%%%%%%%--- load the tsd file to read tsds array ---%%%%%%%%%%%
    addpath('./tsdPackage/')

	%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%

	 try
        sleepScore = importdata(strcat(folderData,'SleepScoring_OBGamma.mat'));
    catch
        sleepScore = importdata(strcat(folderData,'SleepScoring_Accelero.mat'));
    end
	disp('Data Loaded.')

    rem.remStart = Start(sleepScore.("REMEpoch"))/10000;
    rem.remStop = Stop(sleepScore.("REMEpoch"))/10000;

    save(strcat(folderData,'nnREMEpochs.mat'),'rem','-v7.3');
end
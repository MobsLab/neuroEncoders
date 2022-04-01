function extractTsd=extractTsd(folderData)
% the function should receive the folder ending by a /
    %%%%%%%%%%%--- load the tsd file to read tsds array ---%%%%%%%%%%%
    addpath('./tsdPackage/')

	%%%%%%%%%%%--- LOAD NEEDED RESOURCES ---%%%%%%%%%%%

	 try
        SWR = importdata(strcat(folderData,'SWR.mat'));
    catch
        SWR = importdata(strcat(folderData,'ripples.mat'));
    end
	disp('Data Loaded.')

    ripple = SWR.("ripples");

    save(strcat(folderData,'nnSWR.mat'),'ripple','-v7.3');
end
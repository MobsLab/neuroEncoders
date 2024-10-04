function [fh,sq,sweeps, rasterAx, histAx,dArea] = RasterPETH(S, center, TStart, TEnd, varargin)

%% Input
% S is tsd of Spike times
% center is ts of trigger times
% Tstart is (NEGATIVE) value to get time before trigger in ts time scale
% Tend is (POSITIVE) value to get time after trigger in ts time scale
% fix the binsize with the option 'BinSize' in ts time scale

% Output
% fh


font_name = 'Arial';
font_size = 10;
font_weight = 'bold';
line_width = 2;

opt_varargin = varargin;

defined_options  = dictArray({ { 'RasterFraction', {0.7, {'numeric'}} }
    { 'BinSize', {10, {'numeric'}}},
    {'LineWidth', {1, {'numeric'} } },
    {'Markers', { {}, {'cell'}} } ,
    {'MarkerTypes', { {}, {'cell'}}},
    {'MarkerSize', { [], {'numeric'} } },
    });
getOpt;

is = intervalSet(Range(center)+TStart, Range(center)+TEnd);

sweeps = intervalSplit(S, is, 'OffsetStart', TStart);



for iM = 1:length(Markers)
    
    if length(Range(Markers{iM}))==length(Range(center))
        
        Markers{iM} = (Range(Markers{iM}) - Range(center))/10;
    else
        
        %MarkersTemp{iM} =[];
        rgC=Range(center);
        
        for idd=1:length(rgC)
            temp=(Range(Markers{iM})-rgC(idd))/10;
            MarkersTemp{idd} = temp(find(temp>TStart/10&temp<TEnd/10));
            %MarkersTemp{iM} = [MarkersTemp{iM}; temp(find(temp>TStart&temp<TEnd))];
            clear temp
        end
        try
            Markers{iM} = MarkersTemp;
        catch
            
            Markers{iM} =[];
        end
        
    end
    
    
end

rf = RasterFraction * 0.8;
ss = oneSeries(sweeps);
sq = intervalRate(ss, regular_interval(TStart, TEnd, BinSize));
rasterAx = axes('position', [0.1 0.05 0.8 (rf+0.05)]);
histAx = axes('position', [0.1 (rf+0.15) 0.8 (0.75-rf)]);

fh = gcf;
axes(rasterAx);

try
    set(gca, 'FontName', font_name);
    set(gca, 'FontWeight', font_weight);
    set(gca, 'FontSize', font_size);
    set(gca, 'LineWidth', line_width);
    set(gca, 'XLim', [TStart TEnd]/10);
    RasterPlot(sweeps, 'AxHandle', rasterAx, ...
        'FigureHandle', fh, ...
        'TStart', TStart, ...
        'TEnd', TEnd, ...
        'LineWidth', LineWidth, ...
        'Markers', Markers, ...
        'MarkerTypes', MarkerTypes, ...
        'MarkerSize', MarkerSize);
end
set(gca, 'Box', 'on');
axes(histAx);


dArea =  Data(sq)/length(sweeps);
area(Range(sq, 'ms'), Data(sq)/length(sweeps), 'FaceColor', 'k');
set(gca, 'FontName', font_name);
set(gca, 'FontWeight', font_weight);
set(gca, 'FontSize', font_size);
set(gca, 'LineWidth', line_width);
set(gca, 'XLim', [TStart TEnd]/10);
if max(dArea) > 0
    set(gca, 'YLim', [0 max(dArea) * 1.2]);
end
yl = get(gca, 'YTick');
yl = yl(find(yl==floor(yl)));
set(gca, 'YTick', yl);
fh = gcf;
end

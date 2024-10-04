function [fh,sq,sweeps, rasterAx, histAx,dArea] = RasterPETH_order(S, center, TStart, TEnd, varargin)

% [fh,sq,sweeps, rasterAx, histAx,dArea] = RasterPETH_order(S, center, TStart, TEnd, varargin)
% ex:
% figure, [fh,sq,sweeps] = RasterPETH_order(S{i}, ts(End(FreezeEpoch)), -165000,+85000,'BinSize',1000,'Markers',{ts(Start(FreezeEpoch))},'MarkerTypes',{'ro','r'},'order',id);


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
                                {'order', {[],{'double'}}},
                                });
getOpt;

is = intervalSet(Range(center)+TStart, Range(center)+TEnd);

sweeps = intervalSplit(S, is, 'OffsetStart', TStart);
if length(order)>2
    for k=1:length(order)
sweeps2{k}=sweeps{order(k)};
    end
    sweeps = tsdArray(sweeps2);

end

for iM = 1:length(Markers)
    
    if length(Range(Markers{iM}))==length(Range(center))
    
        Markers{iM} = (Range(Markers{iM}) - Range(center))/10; 
        Markersint=Markers{iM};
        Markersint=Markersint(order);
        Markers{iM}=Markersint;
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

ss = oneSeries(sweeps);
sq = intervalRate(ss, regular_interval(TStart, TEnd, BinSize));

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

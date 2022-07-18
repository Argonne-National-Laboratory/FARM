clear; tic
% Specify the file name to read
filename = 'Sweep_Runs_o\sweep\1\out~inner';
% filename = 'Saved_Dispatch_Results\out_inner_FMU';
% filename = 'Saved_Dispatch_Results\out_inner_LTI';
% Specify the interval of x ticks
x_tick_interval=1;  % 2 hours for plot 24-hour result
% x_tick_interval=24; % 24 hours for plot 168-hour result

CGSummary = []; CGSummary_title = {'t', 'Profile', 'r0', 'r1', 'v0', 'v1'};
DispatchSummary=[]; DispatchSummary_title = {'t', 'v0', 'v1', 'y0', 'y0min', 'y0max', 'y1', 'y1min', 'y1max', 'y2', 'y2min', 'y2max', 'y3', 'y3min', 'y3max'};

fid = fopen(filename);
% Get one line from the file
tline = fgetl(fid);
% when this line is not empty (the end of file)
while ischar(tline)
    % collect information for CG summary
    if startsWith(tline, "HaoyuCGSummary,")

        data = nan(1,numel(CGSummary_title)); 
        c = strsplit(tline,',');
        for i=1:numel(CGSummary_title)
            idx_c = find(strcmp(c, CGSummary_title{i}));
            data(i) = str2double(c{idx_c+1});
        end
        CGSummary=[CGSummary;data];

    elseif startsWith(tline, "HaoyuDispatchSummary,")
        data = nan(1,numel(DispatchSummary_title)); 
        c = strsplit(tline,',');
        for i=1:numel(DispatchSummary_title)
            idx_c = find(strcmp(c, DispatchSummary_title{i}));
            data(i) = str2double(c{idx_c+1});
        end
        DispatchSummary=[DispatchSummary;data];        
    end
%     disp(tline)
    tline = fgetl(fid);
end
fclose(fid);
%% Convert time to hours
CGSummary(:,1) = CGSummary(:,1)/3600;
DispatchSummary(:,1) = DispatchSummary(:,1)/3600;

%% convert BOP, SES output power to MW, convert BOP output pressure to bar
for i=[4:6 10:12]
    DispatchSummary(:,i)=DispatchSummary(:,i)*1e-6;
end
for i=7:9
    DispatchSummary(:,i)=DispatchSummary(:,i)*1e-5;
end

%% collect power components
power_provided=DispatchSummary(:,2)+DispatchSummary(:,3);
% power_provided=DispatchSummary(:,4)+DispatchSummary(:,10);
time = DispatchSummary(:,1);
time_array=[];power_array_hour=[];
for i=1:size(DispatchSummary,1)
    if mod(DispatchSummary(i,1),1)==0.5
        time_array = [time_array; DispatchSummary(i,1)];
        power_array_hour=[power_array_hour;DispatchSummary(i,2) DispatchSummary(i,3)];
    end
end

%% 1. Plot the power dispatch stack
% x_label_min=floor(time(1));
x_label_min=0;
x_label_max=ceil(time(end));

figure(10)
set(gcf,'Position',[100 50 600 500])
% Plot the stacked bar of power components
ba = bar(time_array, power_array_hour, 'stacked', 'FaceColor','flat');hold on
ba(1).CData = [0 0.4470 0.7410];
ba(2).CData = [0.9290 0.6940 0.1250];
% Plot the total power provided
plot(time, power_provided,'LineWidth',3,'color','#7E2F8E');hold off
xlabel('Time (Hour)');ylabel('Power (MW)'); 
xlim([x_label_min x_label_max]);xticks(x_label_min:x_tick_interval:x_label_max)
ylim([800 1200])
legend('BOP Output Power','SES Output Power','Market Demand','Location','best')
% legend('BOP','TES Discharging(+)/Charging(-)','Market Demand','Location','best')
title('Contribution of each Power Source')

print('Figure_10_power_contribution.png','-dpng','-r300')

%% 2. Plot the validation history r0, v0 and r1, v1 using CGSummary

% checking the time stamp
NumValidation = numel(find(CGSummary(:,1)==0));
NumHours = size(CGSummary,1)/NumValidation;

for iterValidation = 1:NumValidation
    dispatch_info = CGSummary((iterValidation-1)*NumHours + [1:NumHours],:);
    
    % plot BOP dispatch info
    figure(20) 
    set(gcf,'Position',[100 100 1600 400])
    
    subplot(1,NumValidation,iterValidation)
    hold on
    plot(dispatch_info(:,1), dispatch_info(:,3), 'o', 'MarkerSize',12,'MarkerEdgeColor','#0072BD','MarkerFaceColor','#0072BD')
    plot(dispatch_info(:,1), dispatch_info(:,5), '^', 'MarkerSize',9,'MarkerEdgeColor','#D95319','MarkerFaceColor','#D95319')
    xlabel("Time (Hour)")
    ylabel("BOP Setpoint (MW)")
    ylim([min(CGSummary(:,3)) max(CGSummary(:,3))])
    title(sprintf('FARM Iteration #%d, BOP',iterValidation))
    legend('Setpoint by HERON', 'Setpoint by FARM','Location','northwest')

    % plot SES dispatch info
    figure(30) 
    set(gcf,'Position',[200 200 1600 400])
    
    subplot(1,NumValidation,iterValidation)
    hold on
    plot(dispatch_info(:,1), dispatch_info(:,4), 'o', 'MarkerSize',12,'MarkerEdgeColor','#0072BD','MarkerFaceColor','#0072BD')
    plot(dispatch_info(:,1), dispatch_info(:,6), '^', 'MarkerSize',9,'MarkerEdgeColor','#D95319','MarkerFaceColor','#D95319')
    xlabel("Time (Hour)")
    ylabel("SES Setpoint (MW)")
    ylim([min(CGSummary(:,4)) max(CGSummary(:,4))])
    title(sprintf('FARM Iteration #%d, SES',iterValidation))
    legend('Setpoint by HERON', 'Setpoint by FARM','Location','northwest')
end

% print
figure(20)
print('Figure_20_ValidationHistoryBOP.png','-dpng','-r300')
figure(30)
print('Figure_30_ValidationHistorySES.png','-dpng','-r300')





%% 3. Plot the v0, v1, y0, y1, y2, y3 vs time for "Self Learning Stage" and "Dispatching Stage" 
figure(40)
set(gcf,'Position',[100 100 1800 1200])
FontSize = 12;
% define the logical arrays for learning and dispatching stages
t_learn = time<0; 
% x_learn_min = floor(min(time(t_learn))); 
x_learn_min = -3; 
x_learn_max = ceil(max(time(t_learn)));
t_dispa = time>0; 
x_dispa_min = floor(min(time(t_dispa))); 
x_dispa_max = ceil(max(time(t_dispa)));
% plot the input and outputs during learning and dispatching stages
col_plot=9;
for row_idx=1:6
    % plot self-learning stage
    subplot(6,col_plot,(row_idx-1)*col_plot+[1:2])
    hold on
    if row_idx==1 % power setpoint v0, BOP
        plot(time(t_learn),DispatchSummary(t_learn,2),'Color','#0072BD')
        ylabel('v0, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Self-Learning Stage')
    elseif row_idx==2 % output, y0, BOP Power
        plot(time(t_learn),DispatchSummary(t_learn,6),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,4),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,5),'--b','LineWidth',3)
        ylabel('y0, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, BOP pressure
        plot(time(t_learn),DispatchSummary(t_learn,9),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,7),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,8),'--b','LineWidth',3)
        ylabel('y1, BOP Prs(bar)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, SES
        plot(time(t_learn),DispatchSummary(t_learn,3),'Color','#0072BD')
        ylabel('v1, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, SES Power
        plot(time(t_learn),DispatchSummary(t_learn,12),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,10),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,11),'--b','LineWidth',3)
        ylabel('y2, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #2','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, SES Temperature
        plot(time(t_learn),DispatchSummary(t_learn,15),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,13),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,14),'--b','LineWidth',3)
        ylabel('y3, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,14)); y_ub = max(DispatchSummary(:,15));
        legend('Upper Bound','Output #3','Lower Bound','Location','northeast','FontSize',FontSize)
        xlabel('Time (Hour)','FontSize',FontSize);
    end

    xlim([x_learn_min x_learn_max]);xticks(x_learn_min:x_tick_interval:x_learn_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)

    % plot dispatching stage
    subplot(6,col_plot,(row_idx-1)*col_plot+[4:col_plot])
    hold on
    if row_idx==1 % power setpoint v0, BOP
        plot(time(t_dispa),DispatchSummary(t_dispa,2),'Color','#0072BD')
        ylabel('v0, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Dispatching Stage ')
    elseif row_idx==2 % output, y0, BOP Power
        plot(time(t_dispa),DispatchSummary(t_dispa,6),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,4),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,5),'--b','LineWidth',3)
        ylabel('y0, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, BOP pressure
        plot(time(t_dispa),DispatchSummary(t_dispa,9),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,7),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,8),'--b','LineWidth',3)
        ylabel('y1, BOP Prs(bar)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, SES
        plot(time(t_dispa),DispatchSummary(t_dispa,3),'Color','#0072BD')
        ylabel('v1, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, SES Power
        plot(time(t_dispa),DispatchSummary(t_dispa,12),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,10),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,11),'--b','LineWidth',3)
        ylabel('y2, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, SES Temperature
        plot(time(t_dispa),DispatchSummary(t_dispa,15),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,13),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,14),'--b','LineWidth',3)
        ylabel('y3, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,14)); y_ub = max(DispatchSummary(:,15));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
        xlabel('Time (Hour)','FontSize',FontSize);
    end
    
    xlim([x_dispa_min x_dispa_max]);xticks(x_dispa_min:x_tick_interval:x_dispa_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)
end
print('Figure_40_LearningDispatching_Stage.png','-dpng','-r300')

%% 4. Plot the v0, v1, y0, y1, y2, y3 vs time for "Self Learning Stage" only
figure(50)
set(gcf,'Position',[100 100 1800 1200])
FontSize = 12;
% define the logical arrays for learning and dispatching stages
t_learn = time<0; 
x_learn_min = floor(min(time(t_learn))); 
% x_learn_min = -3; 
x_learn_max = ceil(max(time(t_learn)));

% plot the input and outputs during learning and dispatching stages
for row_idx=1:6
    % plot self-learning stage
    subplot(6,1,row_idx)
    hold on
    if row_idx==1 % power setpoint v0, BOP
        plot(time(t_learn),DispatchSummary(t_learn,2),'Color','#0072BD')
        ylabel('v0, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Self-Learning Stage')
    elseif row_idx==2 % output, y0, BOP Power
        plot(time(t_learn),DispatchSummary(t_learn,6),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,4),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,5),'--b','LineWidth',3)
        ylabel('y0, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, BOP pressure
        plot(time(t_learn),DispatchSummary(t_learn,9),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,7),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,8),'--b','LineWidth',3)
        ylabel('y1, BOP Prs(bar)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, SES
        plot(time(t_learn),DispatchSummary(t_learn,3),'Color','#0072BD')
        ylabel('v1, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, SES Power
        plot(time(t_learn),DispatchSummary(t_learn,12),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,10),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,11),'--b','LineWidth',3)
        ylabel('y2, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #2','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, SES Temperature
        plot(time(t_learn),DispatchSummary(t_learn,15),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,13),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,14),'--b','LineWidth',3)
        ylabel('y3, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,14)); y_ub = max(DispatchSummary(:,15));
        legend('Upper Bound','Output #3','Lower Bound','Location','northeast','FontSize',FontSize)
        xlabel('Time (Hour)','FontSize',FontSize);
    end

    xlim([x_learn_min x_learn_max]);xticks(x_learn_min:x_tick_interval:x_learn_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)

end
print('Figure_50_Learning_Stage.png','-dpng','-r300')

%% 5. Plot the v0, v1, y0, y1, y2, y3 vs time for  "Dispatching Stage" only
figure(60)
set(gcf,'Position',[100 100 1500 1200])
FontSize = 12;
% define the logical arrays for dispatching stages
t_dispa = time>0; 
x_dispa_min = floor(min(time(t_dispa))); 
x_dispa_max = ceil(max(time(t_dispa)));
% plot the input and outputs during learning and dispatching stages

for row_idx=1:6
    % plot self-learning stage
    subplot(6,1,row_idx)
    hold on

    if row_idx==1 % power setpoint v0, BOP
        plot(time(t_dispa),DispatchSummary(t_dispa,2),'Color','#0072BD')
        ylabel('v0, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Dispatching Stage ')
    elseif row_idx==2 % output, y0, BOP Power
        plot(time(t_dispa),DispatchSummary(t_dispa,6),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,4),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,5),'--b','LineWidth',3)
        ylabel('y0, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, BOP pressure
        plot(time(t_dispa),DispatchSummary(t_dispa,9),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,7),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,8),'--b','LineWidth',3)
        ylabel('y1, BOP Prs(bar)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, SES
        plot(time(t_dispa),DispatchSummary(t_dispa,3),'Color','#0072BD')
        ylabel('v1, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, SES Power
        plot(time(t_dispa),DispatchSummary(t_dispa,12),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,10),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,11),'--b','LineWidth',3)
        ylabel('y2, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, SES Temperature
        plot(time(t_dispa),DispatchSummary(t_dispa,15),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,13),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,14),'--b','LineWidth',3)
        ylabel('y3, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,14)); y_ub = max(DispatchSummary(:,15));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
        xlabel('Time (Hour)','FontSize',FontSize);
    end
    
    xlim([x_dispa_min x_dispa_max]);xticks(x_dispa_min:x_tick_interval:x_dispa_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)
end
print('Figure_60_Dispatching_Stage.png','-dpng','-r300')

%%
toc

clear; tic
% Specify the file name to read
filename = 'Sweep_Runs_o\sweep\1\out~inner';
% filename = 'Saved_Dispatch_Results\out_inner_FMU';
% filename = 'Saved_Dispatch_Results\out_inner_LTI';

% Specify the interval of x ticks
x_tick_interval=1;  % 2 hours for plot 24-hour result
% x_tick_interval=24; % 24 hours for plot 168-hour result

% Specify the number of input, state and output variables 
Num_r = 2; % SES_Demand_MW, BOP_Demand_MW
Num_x = 2; % SES.GTunit.combChamber.fluegas.h, BOP.CS.PID_TCV_opening.addP.u2
Num_y = 4; % SES_Electric_Power, SES_Firing_Temperature, BOP_Electric_Power, BOP_Turbine_Pressure

CGSummary = []; CGSummary_title = {'t', 'Profile', 'r', 'v'};
DispatchSummary=[]; DispatchSummary_title = {'t', 'v', 'y', 'ymin', 'ymax'};
vxSummary=[]; vxSummary_title = {'t', 'v', 'x'};

fid = fopen(filename);
% Get one line from the file
tline = fgetl(fid);
% when this line is not empty (the end of file)
while ischar(tline)
    % collect information for CG summary
    if startsWith(tline, "HaoyuCGSummary,")
        data = nan(1,2+Num_r+Num_r); % 1-t; 2-Profile; 3&4-r; 5&6-v;
        c = strsplit(tline,', ');
        for i=1:numel(c)
            if strcmp(c{i},'t') % extract time
                data(1) = str2double(c{i+1});
            elseif strcmp(c{i},'Profile') % extract profile
                data(2) = str2double(c{i+1});    
            elseif strcmp(c{i},'r') % extract vector r
                for j = 1:Num_r
                    data(2+j) = str2double(c{i+j});
                end
            elseif strcmp(c{i},'v') % extract vector v
                for j = 1:Num_r
                    data(2+Num_r+j) = str2double(c{i+j});
                end
            end
        end
        CGSummary=[CGSummary;data];

    % collect information for dispatch summary
    elseif startsWith(tline, "HaoyuDispatchSummary,")
        datadp = nan(1,1+Num_r+3*Num_y); 
        datavx = nan(1,1+Num_r+Num_x); 
        c = strsplit(tline,', ');
        for i=1:numel(c)
            if strcmp(c{i},'t') % extract time
                datadp(1) = str2double(c{i+1});
            elseif strcmp(c{i},'v') % extract vector v
                for j = 1:Num_r
                    datadp(1+j) = str2double(c{i+j});
                end
            elseif strcmp(c{i},'y') % extract vector y
                for j = 1:Num_y
                    datadp(1+Num_r+j) = str2double(c{i+j});
                end
            elseif strcmp(c{i},'ymin') % extract vector ymin
                for j = 1:Num_y
                    datadp(1+Num_r+Num_y+j) = str2double(c{i+j});
                end
            elseif strcmp(c{i},'ymax') % extract vector y
                for j = 1:Num_y
                    datadp(1+Num_r+2*Num_y+j) = str2double(c{i+j});
                end
            end
        end
        DispatchSummary=[DispatchSummary;datadp];        
        
        for i=1:numel(c)
            if strcmp(c{i},'t') % extract time
                datavx(1) = str2double(c{i+1});
            elseif strcmp(c{i},'v') % extract vector v
                for j = 1:Num_r
                    datavx(1+j) = str2double(c{i+j});
                end
            elseif strcmp(c{i},'x') % extract vector x
                for j = 1:Num_x
                    datavx(1+Num_r+j) = str2double(c{i+j});
                end
            end
        end
        vxSummary=[vxSummary;datavx];        
    end
%     disp(tline)
    tline = fgetl(fid);
end
fclose(fid);
%% find the unique time (HERON may run multiple times)
time_unique = unique(DispatchSummary(:,1));
num_unique_entries = numel(time_unique);
DispatchSummary(num_unique_entries+1:end,:)=[];
vxSummary(num_unique_entries+1:end,:)=[];

%% Convert time to hours
CGSummary(:,1) = CGSummary(:,1)/3600;
DispatchSummary(:,1) = DispatchSummary(:,1)/3600;
vxSummary(:,1) = vxSummary(:,1)/3600;

%% convert BOP, SES output power (y1, y3)to MW, convert BOP output pressure (y4) to bar
for i=4:2:14
    DispatchSummary(:,i)=DispatchSummary(:,i)*1e-6;
end
for i=7:4:15
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
        power_array_hour=[power_array_hour;DispatchSummary(i,3) DispatchSummary(i,2)];
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

%% 2. TODO: Plot the validation history r0, v0 and r1, v1 using CGSummary

% checking the time stamp
NumValidation = numel(find(CGSummary(:,1)==0));
NumHours = size(CGSummary,1)/NumValidation;

for iterValidation = 1:NumValidation
    dispatch_info = CGSummary((iterValidation-1)*NumHours + [1:NumHours],:);
    
    % plot SES dispatch info
    figure(20) 
    set(gcf,'Position',[100 100 1600 400])
    
    subplot(1,NumValidation,iterValidation)
    hold on
    plot(dispatch_info(:,1), dispatch_info(:,3), 'o', 'MarkerSize',12,'MarkerEdgeColor','#0072BD','MarkerFaceColor','#0072BD')
    plot(dispatch_info(:,1), dispatch_info(:,5), '^', 'MarkerSize',9,'MarkerEdgeColor','#D95319','MarkerFaceColor','#D95319')
    xlabel("Time (Hour)")
    ylabel("SES Setpoint (MW)")
    ylim([min(CGSummary(:,3)) max(CGSummary(:,3))])
    title(sprintf('FARM Iteration #%d, SES',iterValidation))
    legend('Setpoint by HERON', 'Setpoint by FARM','Location','northwest')

    % plot BOP dispatch info
    figure(30) 
    set(gcf,'Position',[200 200 1600 400])
    
    subplot(1,NumValidation,iterValidation)
    hold on
    plot(dispatch_info(:,1), dispatch_info(:,4), 'o', 'MarkerSize',12,'MarkerEdgeColor','#0072BD','MarkerFaceColor','#0072BD')
    plot(dispatch_info(:,1), dispatch_info(:,6), '^', 'MarkerSize',9,'MarkerEdgeColor','#D95319','MarkerFaceColor','#D95319')
    xlabel("Time (Hour)")
    ylabel("BOP Setpoint (MW)")
    ylim([min(CGSummary(:,4)) max(CGSummary(:,4))])
    title(sprintf('FARM Iteration #%d, BOP',iterValidation))
    legend('Setpoint by HERON', 'Setpoint by FARM','Location','northwest')
end

% print
figure(20)
print('Figure_20_ValidationHistorySES.png','-dpng','-r300')
figure(30)
print('Figure_30_ValidationHistoryBOP.png','-dpng','-r300')





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
    if row_idx==1 % power setpoint v0, SES
        plot(time(t_learn),DispatchSummary(t_learn,2),'Color','#0072BD')
        ylabel('v0, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Self-Learning Stage')
    elseif row_idx==2 % output, y0, SES Power
        plot(time(t_learn),DispatchSummary(t_learn,6),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,4),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,5),'--b','LineWidth',3)
        ylabel('y0, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, SES Temperature
        plot(time(t_learn),DispatchSummary(t_learn,9),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,7),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,8),'--b','LineWidth',3)
        ylabel('y1, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, BOP
        plot(time(t_learn),DispatchSummary(t_learn,3),'Color','#0072BD')
        ylabel('v1, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, BOP Power
        plot(time(t_learn),DispatchSummary(t_learn,12),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,10),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,11),'--b','LineWidth',3)
        ylabel('y2, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #2','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, BOP pressure
        plot(time(t_learn),DispatchSummary(t_learn,15),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,13),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,14),'--b','LineWidth',3)
        ylabel('y3, BOP Prs(bar)','FontSize',FontSize);
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
    if row_idx==1 % power setpoint v0, SES
        plot(time(t_dispa),DispatchSummary(t_dispa,2),'Color','#0072BD')
        ylabel('v0, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Dispatching Stage')
    elseif row_idx==2 % output, y0, SES Power
        plot(time(t_dispa),DispatchSummary(t_dispa,6),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,4),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,5),'--b','LineWidth',3)
        ylabel('y0, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, SES Temperature
        plot(time(t_dispa),DispatchSummary(t_dispa,9),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,7),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,8),'--b','LineWidth',3)
        ylabel('y1, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, BOP
        plot(time(t_dispa),DispatchSummary(t_dispa,3),'Color','#0072BD')
        ylabel('v1, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, BOP Power
        plot(time(t_dispa),DispatchSummary(t_dispa,12),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,10),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,11),'--b','LineWidth',3)
        ylabel('y2, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #2','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, BOP pressure
        plot(time(t_dispa),DispatchSummary(t_dispa,15),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,13),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,14),'--b','LineWidth',3)
        ylabel('y3, BOP Prs(bar)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,14)); y_ub = max(DispatchSummary(:,15));
        legend('Upper Bound','Output #3','Lower Bound','Location','northeast','FontSize',FontSize)
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
    if row_idx==1 % power setpoint v0, SES
        plot(time(t_learn),DispatchSummary(t_learn,2),'Color','#0072BD')
        ylabel('v0, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Self-Learning Stage')
    elseif row_idx==2 % output, y0, SES Power
        plot(time(t_learn),DispatchSummary(t_learn,6),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,4),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,5),'--b','LineWidth',3)
        ylabel('y0, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, SES Temperature
        plot(time(t_learn),DispatchSummary(t_learn,9),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,7),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,8),'--b','LineWidth',3)
        ylabel('y1, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, BOP
        plot(time(t_learn),DispatchSummary(t_learn,3),'Color','#0072BD')
        ylabel('v1, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, BOP Power
        plot(time(t_learn),DispatchSummary(t_learn,12),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,10),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,11),'--b','LineWidth',3)
        ylabel('y2, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #2','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, BOP pressure
        plot(time(t_learn),DispatchSummary(t_learn,15),'--r','LineWidth',3)
        plot(time(t_learn),DispatchSummary(t_learn,13),'-k')
        plot(time(t_learn),DispatchSummary(t_learn,14),'--b','LineWidth',3)
        ylabel('y3, BOP Prs(bar)','FontSize',FontSize);
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

    if row_idx==1 % power setpoint v0, SES
        plot(time(t_dispa),DispatchSummary(t_dispa,2),'Color','#0072BD')
        ylabel('v0, SES SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,2)); y_ub = max(DispatchSummary(:,2));
        title('Dispatching Stage')
    elseif row_idx==2 % output, y0, SES Power
        plot(time(t_dispa),DispatchSummary(t_dispa,6),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,4),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,5),'--b','LineWidth',3)
        ylabel('y0, SES Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,5)); y_ub = max(DispatchSummary(:,6));
        legend('Upper Bound','Output #0','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==3 % output, y1, SES Temperature
        plot(time(t_dispa),DispatchSummary(t_dispa,9),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,7),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,8),'--b','LineWidth',3)
        ylabel('y1, SES Temp.(K)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,8)); y_ub = max(DispatchSummary(:,9));
        legend('Upper Bound','Output #1','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==4 % power setpoint v1, BOP
        plot(time(t_dispa),DispatchSummary(t_dispa,3),'Color','#0072BD')
        ylabel('v1, BOP SetPnt(MW)','FontSize',FontSize); 
        y_lb = min(DispatchSummary(:,3)); y_ub = max(DispatchSummary(:,3));
    elseif row_idx==5 % output, y2, BOP Power
        plot(time(t_dispa),DispatchSummary(t_dispa,12),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,10),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,11),'--b','LineWidth',3)
        ylabel('y2, BOP Pwr(MW)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,11)); y_ub = max(DispatchSummary(:,12));
        legend('Upper Bound','Output #2','Lower Bound','Location','northeast','FontSize',FontSize)
    elseif row_idx==6 % output, y3, BOP pressure
        plot(time(t_dispa),DispatchSummary(t_dispa,15),'--r','LineWidth',3)
        plot(time(t_dispa),DispatchSummary(t_dispa,13),'-k')
        plot(time(t_dispa),DispatchSummary(t_dispa,14),'--b','LineWidth',3)
        ylabel('y3, BOP Prs(bar)','FontSize',FontSize);
        y_lb = min(DispatchSummary(:,14)); y_ub = max(DispatchSummary(:,15));
        legend('Upper Bound','Output #3','Lower Bound','Location','northeast','FontSize',FontSize)
        xlabel('Time (Hour)','FontSize',FontSize);
    end
    
    xlim([x_dispa_min x_dispa_max]);xticks(x_dispa_min:x_tick_interval:x_dispa_max)
    ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
    ytickformat('%.1f')
    set(gca,'FontSize',FontSize)
end
print('Figure_60_Dispatching_Stage.png','-dpng','-r300')

%% Plot the v0, v1, x0, x1 shift history, around t=-4 or t=-3
t_idx = find(vxSummary(:,1)==-4);
plot_x_width = 5;

figure(70)
set(gcf,'Position',[100 100 1600 800])
FontSize = 12;
col_plot = 3;
row_plot = 4;
for row_id = 1:row_plot
    for col_id = 1:col_plot
        subplot(row_plot,col_plot,(row_id-1)*col_plot+col_id)
        
        if row_id == 1
            stairs([-plot_x_width:plot_x_width]*10, vxSummary((t_idx-plot_x_width:t_idx+plot_x_width),row_id+1), 'LineWidth',3)
            y_ub = max(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width),row_id+1));
            y_lb = min(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width),row_id+1));
            ylabel('v0')
            legend('SES SetPoint(MW)','Location','east')
            title(sprintf("Input Shift = %.1d time steps",(col_id-1)))
        elseif row_id == 2
            stairs([-plot_x_width:plot_x_width]*10, vxSummary((t_idx-plot_x_width:t_idx+plot_x_width)-col_id+3,row_id+1), 'LineWidth',3)
            y_ub = max(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width)-col_id+3,row_id+1));
            y_lb = min(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width)-col_id+3,row_id+1));
            ylabel('v1')
            legend('BOP SetPoint(MW)','Location','east')
        elseif row_id == 3
            stairs([-plot_x_width:plot_x_width]*10, vxSummary((t_idx-plot_x_width:t_idx+plot_x_width),row_id+1), 'LineWidth',3, 'Color','#D95319')
            y_ub = max(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width),row_id+1));
            y_lb = min(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width),row_id+1));
            ylabel('x0')
            legend('SES.GTunit.combChamber.fluegas.h(J/kg)','Location','east')
        elseif row_id == 4
            stairs([-plot_x_width:plot_x_width]*10, vxSummary((t_idx-plot_x_width:t_idx+plot_x_width)-col_id+3,row_id+1), 'LineWidth',3, 'Color','#D95319')
            y_ub = max(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width)-col_id+3,row_id+1));
            y_lb = min(vxSummary((t_idx-plot_x_width:t_idx+plot_x_width)-col_id+3,row_id+1));
            ylabel('x1')
            xlabel('Time (seconds)')
            legend('BOP.CS.PID\_TCV\_opening.addP.u2(W)','Location','east')
        end
        xticks([-plot_x_width:plot_x_width]*10)
        ylim([y_lb-(y_ub-y_lb)*0.2 y_ub+(y_ub-y_lb)*0.2])
        set(gca,'FontSize',FontSize)
    end

%     set(gca,'FontSize',FontSize)


end
print('Figure_70_Input_Shift.png','-dpng','-r300')


% <InputVarNames> SES_Demand_MW, BOP_Demand_MW </InputVarNames>
% <StateVarNames> SES.GTunit.combChamber.fluegas.h, BOP.CS.PID_TCV_opening.addP.u2 </StateVarNames>
% <OutputVarNames> SES_Electric_Power, SES_Firing_Temperature, BOP_Electric_Power, BOP_Turbine_Pressure </OutputVarNames>

%%
toc

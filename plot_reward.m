% figure
% hold on;
task = 'hflip';
subfolder = {'TNN_layer_1', 'TNN_layer_2', 'TNN_layer_3', 'TNN_layer_4', 'TNN_layer_5'};
time = cell(6,1);
meanreward100episode = cell(6,1);
[time{1},meanreward100episode{1},~,~,~,~] = ...
    importfile('original/train_log_20170306-025027.txt');
for i=1:5
    filename_to_find = [task, '/', subfolder{i}, '/train_log_*'];
    MyFolderInfo = dir(filename_to_find);
    [time{i+1},meanreward100episode{i+1},~,~,~,~] = ...
        importfile([MyFolderInfo.folder,'/',MyFolderInfo.name]);
    
end

%%
figure
hold on;
for i=1:6
    plot(time{i},meanreward100episode{i},...
        'Linewidth', 2);
end

set(gca, 'Fontsize', 20)
box on
xlim([0, 2e6]);
ylim([-25, 20]);
xlabel('Env Timesteps')
ylabel({'Mean Reward of', 'recent 100 episode'})
title('Pong game and Hflip variation')
legend({'Original', 'Hflip-branch@1',...
    'Hflip-branch@2', 'Hflip-branch@3',...
    'Hflip-branch@4', 'Hflip-branch@5'}, 'Location', 'east')
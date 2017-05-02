% figure
% hold on;
task = 'vflip';
subfolder = {'TNN_Layer_1', 'TNN_Layer_3', 'TNN_Layer_5'};
time = cell(4,1);
meanreward100episode = cell(4,1);
[time{1},meanreward100episode{1},~,~,~,~] = ...
    importfile('original/train_log_20170306-025027.txt');
for i=1:3
    filename_to_find = [task, '/', subfolder{i}, '/train_log_*'];
    MyFolderInfo = dir(filename_to_find);
    [time{i+1},meanreward100episode{i+1},~,~,~,~] = ...
        importfile([MyFolderInfo.folder,'/',MyFolderInfo.name]);
    
end

%%
figure
hold on;
for i=1:4
    plot(time{i},meanreward100episode{i},...
        'Linewidth', 2);
end

set(gca, 'Fontsize', 20)
box on
xlim([0, 2e6]);
ylim([-25, 20]);
xlabel('Env Timesteps')
ylabel({'Mean Reward of', 'recent 100 episode'})
title('Pong game and Vflip variation')
legend({'Original', 'Vflip-branch@1',...
     'Vflip-branch@3', 'Vflip-branch@5'}, 'Location', 'east')
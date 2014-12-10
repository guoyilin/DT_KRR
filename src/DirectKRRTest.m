%% Test for direct KRR method.

%% read data from file
% meta_file = 'data/meta';
% meta_data = load(meta_file);
% file1 = dir('data/n04429376/*.mat');
% file2 = dir('data/n13044778/*.mat');
% trainNum = length(file1) + length(file2);
% TrainData = zeros(trainNum, 4096); 
% TrainLabels = ones(trainNum, 2) * (-1);
% 
% count = 1
% for file = file1'
%     image = load(strcat('data/n04429376/',file.name));
%     TrainData(count, :) = image.data;
%     TrainLabels(count, 1) = 1;
%     count= count +1;
% end
% disp(count);
% for file = file2'
%     image = load(strcat('data/n13044778/',file.name));
%     TrainData(count, :) = image.data;
%     TrainLabels(count, 2) = 1;
%     count= count +1;
% end
load('data/10.mat');   
%% normalize data to [0,1]
train_minimums = min(TrainData, [], 2);
train_ranges = max(TrainData, [], 2) - train_minimums;
TrainData = (TrainData - repmat(train_minimums, 1, size(TrainData, 2))) ./ repmat(train_ranges, 1, size(TrainData, 2));
test_minimums = min(TestData, [], 2);
test_ranges = max(TestData, [], 2) - test_minimums;
TestData = (TestData - repmat(test_minimums, 1, size(TestData, 2))) ./ repmat(test_ranges, 1, size(TestData, 2));

%% train use direct krr.
 gamma = stdv(TrainData); 
 [Alpha, train_data, kernel_gamma] = DirectKRR_Representer(TrainData,TrainLabels,gamma,0.1);
 Values = DirectKRRPredict(Alpha, train_data, kernel_gamma, TestData, TestLabels);
 accuracy = ComputeAccuracy(Values, TestLabels);
 disp(['accuracy:' num2str(accuracy)]);
% average squared distance
%for i = 3:10
gamma = 4096;
% for i=-10:10
%     lambda = 2^i;
%      [Alpha, train_data, kernel_gamma] = NystromSampling(gamma,TrainData,TrainLabels,500, 'r', lambda);
%      Values = DirectKRRPredict(Alpha, train_data, kernel_gamma, TestData, TestLabels);
%     accuracy = ComputeAccuracy(Values, TestLabels);
%     disp(['accuracy:' num2str(accuracy)]);
% end
% [Alpha, train_data, kernel_gamma] = NystromSampling(gamma,TrainData,TrainLabels,500 , 'k', 0.1);
%  Values = DirectKRRPredict(Alpha, train_data, kernel_gamma, TestData, TestLabels);
% accuracy = ComputeAccuracy(Values, TestLabels);
% disp(['accuracy:' num2str(accuracy)]);


%end


function [ output_args ] = main( input_args )
global TrainData;
global TrainLabels;
global TestData;
global TestLabels;
global globalattriNum;
%% read data from file
%meta_file = 'data/meta';
%meta_data = load(meta_file);
%dict = make_hash(meta_data.synsets);
if (exist('data/50.mat', 'file') ==0)
   train_dir = dir('/mnt/data/imageNet2012/decafCentre/train/matlab_50/train/n*');
test_dir = dir('/mnt/data/imageNet2012/decafCentre/train/matlab_50/test/n*');
dict = java.util.Hashtable;
count = 1;
for file=train_dir'
   dict.put(file.name, count);
   count = count + 1;
end
train_count = 1;
for file=train_dir'
    category_name = file.name;
    filename = strcat('/mnt/data/imageNet2012/decafCentre/train/matlab_50/train/', file.name);
    image_dir = dir([filename, '/*.mat']);
    for image=image_dir'
        image = load([filename, '/', image.name]);
        TrainData(train_count, :) = image.data;
        TrainLabels(train_count, :) = ones(1, size(train_dir, 1)) * (-1);
        TrainLabels(train_count, dict.get(category_name)) = 1;
        train_count = train_count + 1;
    end
end
test_count = 1;
for file=test_dir'
    category_name = file.name;
    filename = strcat('/mnt/data/imageNet2012/decafCentre/train/matlab_50/test/', file.name);
    image_dir = dir([filename, '/*.mat']);
    for image=image_dir'
        image = load([filename, '/', image.name]);
        TestData(test_count, :) = image.data;
        TestLabels(test_count, :) = ones(1, size(test_dir, 1)) * (-1);
        TestLabels(test_count, dict.get(category_name)) = 1;
        test_count = test_count + 1;
    end
end 
else
load('data/50.mat');   
end
%% we save it into data/matlab.mat
%% clear unuse.
%% normalize data to [0,1]
train_minimums = min(TrainData, [], 2);
train_ranges = max(TrainData, [], 2) - train_minimums;
TrainData = (TrainData - repmat(train_minimums, 1, size(TrainData, 2))) ./ repmat(train_ranges, 1, size(TrainData, 2));
test_minimums = min(TestData, [], 2);
test_ranges = max(TestData, [], 2) - test_minimums;
TestData = (TestData - repmat(test_minimums, 1, size(TestData, 2))) ./ repmat(test_ranges, 1, size(TestData, 2));
globalattriNum = size(TrainData, 2);
%% train using Nystrom method.
% gamma = globalattriNum;
% TrainIndex = linspace(1, size(TrainData, 1), size(TrainData, 1));
% TestIndex = linspace(1, size(TestData, 1), size(TestData, 1));
% [Alpha, train_index, kernel_gamma]  = NystromSampling( gamma,TrainIndex', 0.1 * size(TrainData, 1), 'r', 2^-4 );
% Values = DirectKRRPredict(Alpha, train_index, kernel_gamma, TestIndex');
% accuracy = ComputeAccuracy(Values, TestLabels);
% disp(accuracy);
% % save 'record/Nystrom0.1r.mat' accuracy
% clear Alpha;
% clear Values;
% [Alpha, train_index, kernel_gamma]  = NystromSampling( gamma,TrainIndex', 0.1 * size(TrainData, 1), 'k', 2^-4 );
%  Values = DirectKRRPredict(Alpha, train_index, kernel_gamma,  TestIndex');
%  accuracy = ComputeAccuracy(Values, TestLabels);
% disp(accuracy);
%save 'record/Nystrom0.1k.mat' accuracy


%% training using hierarchical krr.
% set parameter.
savematrix=zeros(1,4);
count = 1;
gamma = globalattriNum;% we use rbf kernel.
sampleStrategy = 'r'; % Nystrom-sampling stategy: k-means, random.
for num=1:5
for i =-4
lambda =2^i;
%disp(['splitNum:', num2str(i * 100)]);
disp('train start...');
TrainIndex = linspace(1, size(TrainData, 1), size(TrainData, 1));
Node = decisionTree( TrainIndex', 1000*num, gamma, lambda, sampleStrategy, 0, 0.1);
disp('predict start...');
TestIndex = linspace(1, size(TestData, 1), size(TestData, 1));
accuracy = classify(TestIndex', Node);
disp(accuracy / size(TestLabels, 1));
%save result
savematrix(count,1)=1000*num;
savematrix(count,2)=gamma;
savematrix(count,3)=lambda;
savematrix(count,4)=accuracy / size(TestLabels, 1);
count = count + 1;
end
end
save 'record/result.mat' savematrix;
end


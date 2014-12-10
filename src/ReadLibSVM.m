function [ output_args ] = ReadLibSVM( input_args )
%READLIBSVM Summary of this function goes here
%   Detailed explanation goes here
[TrainLabels, TrainData] = libsvmread('data/10_train.txt');
[TestLabels, TestData] = libsvmread('data/10_test.txt');
% change the TrainLabels according to dt_test.m format. and use dt_test.m
% to run.
end


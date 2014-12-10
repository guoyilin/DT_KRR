function accuracy = ComputeAccuracy( Predict_Values, Labels )
%COMPUTEACCURACY Summary of this function goes here
%   Detailed explanation goes here
[~, I1 ] = max(Predict_Values');
[~, I2] = max(Labels');
index = I1 == I2;
accuracy = size(find(index == 1), 2);
accuracy = accuracy / size(Predict_Values, 1);
end


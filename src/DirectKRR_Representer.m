function [Alpha, train_data, kernel_gamma] = DirectKRR_Representer(data,Y,gamma,lambda)
%
% This function performs the kernel ridge regression using the Gaussian
% Kernel. Anyother kernel can be used but according to Mercers Theorem it
% should not matter too much.
%
% FinalAns = KernelRidge(In_Data,Out_Data,Test_Data,Lamda)
%
% in_data - Input to the functio to be regressed. D (dimensional) X N (points)
% out_data - Ouput of the function to be regressed. T X N (points), T is
% the category number. for i=1,...,T, If Ti = 1, i is the label. else Ti = -1.
% test_data - Input not included in training. D (dimensions) X n (points)
% lamda - For tikhonov regularization. (Carefully choose this)
% final_ans - Output for a new set of inputs (those that were not in
%             training) 1 X n (points)  
% gamma - the gamma for gaussian.
% for linear ridge regression use the matlab function "ridge"
% Author - Ambarish Jash & Yilin Guo.
% ref - http://www.eecs.berkeley.edu/~wainwrig/stat241b/lec6.pdf

if size(data,1) ~= size(Y,1)
    fprintf('\n number of training data and number of Y is not equal');
    fprintf('\n Exitting program');
    return
else
    train_data = data;
    gamma = stdv(data);
   lambda = 1 / (gamma * size(data, 1));
    kernel_gamma = gamma;
  %  disp(['gamma:', num2str(gamma)]);
   % disp(['lambda:', num2str(lambda)]);
    %% construction of kernel matrix.
%    Kernel = zeros(size(data,1),size(data,1));
    % x_in(i,j) = x_in(j,i) -- Using symmetry of the Kernel%
 %   for row = 1:size(Kernel,2)
 %       for col = 1:row
 %           temp = sum((data(:,row)-data(:,col)).^2);
 %           Kernel(row,col) = exp(-gamma * temp);
 %       end
 %   end
 %   Kernel = Kernel + Kernel';
 %   for count = 1:size(Kernel,2)
 %       Kernel(count,count) = Kernel(count,count)/2;
 %   end
Kernel = exp(- sqdist(train_data', train_data')/gamma);
    %% Calculating alpha
   % if det(Kernel + lamda*eye(size(Kernel))) > 1e9
   %     fprintf('\nThe kernel matrix is poorly scaled. Please choose a better scaling parameter.');
   %     return
   % end
Alpha = inv(Kernel + lambda*eye(size(Kernel)))*Y; % N * T
    
end


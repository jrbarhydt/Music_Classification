function [proj,U,w,thresh_line] = LDA_train(band1,band2,num_modes)
%This function was designed to accept two bands' spectrogram sets and
%truncation rank, to perform LDA.
% Arguments:
%-----------------------------
%     band1: vectorized grouping of training set for band#1
%     band2: vectorized grouping of training set for band#2
% num_modes: number of LDA features to keep after projection
%    
%          
% Returns:
%-----------------------------
%        proj: projection vector of bands onto LDA basis
%           U: truncated (by num_modes) SVD basis modes. Use U'*test to
%              create test set SVD projection.
%           w: truncated LDA feature space. Use w'*U'*test to project test
%              set onto LDA basis for discrimination.
% thresh_line: descriminator value
%--------------------------------------------------------------------------
    % default arguments
    if nargin < 2
       down_rate = 10; 
    end
    if nargin < 3
       time_steps = 20; 
    end
    if nargin < 4
        gabor_w = 1000;
    end
%band bundling
num_1=size(band1,2);num_2=size(band2,2);
band_concat=[band1,band2];
[U,S,V]=svd(band_concat,'econ');
%bands=S*V'; %band features in svd basis
U = U(:,1:num_modes);
band1_group = bands(1:num_modes,1:num_1);
band2_group = bands(1:num_modes,num_1+1:num_1+num_2);

m1=mean(band1_group,2);m2=mean(band2_group,2);
Sw=0; %diagonal variances
for i=1:num_1
    variance = (band1_group(:,i)-m1);
    Sw=Sw+variance*variance';
end
for i=1:num_2
    variance = (band2_group(:,i)-m2);
    Sw=Sw+variance*variance';
end
Sb=(m1-m2)*(m1-m2)'; %cross variances

[V2,D]=eig(Sb,Sw); %LDA
[~,ind]=max(abs(diag(D))); %feature truncation
w=V2(:,ind)/norm(V2(:,ind));

vband1= w'*band1_group; vband2= w'*band2_group;
proj=[vband1, vband2];

% linear sorting, classification
if mean(vband1)>mean(vband2)
    w=-w;vband1=-vband1;vband2=-vband2;
end
band1_dec=sort(vband1);
band2_dec=sort(vband2);
t1=length(band1_dec);t2=1;
while band1_dec(t1)>band2_dec(t2)
    t1=t1-1; %dec t1
    t2=t2+1; %inc t2
end

thresh_line=(band1_dec(t1)+band2_dec(t2))/2;
end


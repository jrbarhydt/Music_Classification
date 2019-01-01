% Music Genre Identification: Spectrogram Analysis, Classification
% Johnathon R Barhydt
%
% To get quality spectrograms, tune up the inputs to sgram, default will
% be quick. Currently file heirarchy must be maintained, and each band
% is to have equal number of songs. This can be easily changed.
%
% specs for this run:
% 3 5sec clips per song
% 16 songs per band
% 3 bands per genre
% 3 genres
%-----------------
% total of 432 clips
%
clear all; close all; clc

% root music directory, must have genre>band>song substructure
directory='C:\\Music';
% projection rank, number of spectrogram basis modes in discrimination
rank=5;
% num x-validation iterations
perms=500;
% generate genre lists / band lists / song lists
main_dir=dir(directory);
% build genre list
genre_list=strings(length(main_dir)-2,1);
for i=1:length(genre_list)
   genre_list(i)=string( main_dir(i+2).name); 
end
% build band list
for g=1:length(genre_list)
    genre_dir_path= char(strcat(directory,"\",genre_list( g )));
    genre_dir=dir(genre_dir_path);
    band_list=strings( length( genre_dir)-2, 1);
    for i=1:length(band_list)
       band_list(i)=string( genre_dir( i+2).name); 
    end
    % build song list
    for b=1:length(genre_list)
        band_dir_path = char(strcat(directory,"\",genre_list( g ),"\",band_list( b )));
        band_dir=dir(band_dir_path);
        song_list=strings( length( band_dir)-2, 1);
        for i=1:length(song_list)
           song_list(i)=string( band_dir( i+2).name); 
        end

        % generates spectrograms of each song clip (CAN BE VERY COSTLY!)
        for s=1:length(song_list) 
            song_path = char(strcat(directory,"\",genre_list( g ),"\",band_list( b ),"\",song_list( s )));
            % sgram(downsample_rate,number_timesteps,gab_window_narrowness"
            sg1(:,:,s)=sgram(song_path, 10, 20, 1000);
            sg2(:,:,s)=sgram(song_path, 10, 20, 1000);
            sg3(:,:,s)=sgram(song_path, 10, 20, 1000);
        end

        % export spectrograms as Genre_BandName.mat
        filename = strcat(genre_list(g),'_',band_list(b),'.mat');
        % collect sgrams
        sg = cat(3,sg1,sg2,sg3);
        save(filename, 'sg');
    end
end

%% Load Saved Spectrograms For Analysis
%
A=importdata('Classical_Beethoven.mat');
sp_w=size(A,1);sp_h=size(A,2);
for i=1:size(A,3)
    beeth(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Classical_Mozart.mat');
for i=1:size(A,3)
    mozrt(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Classical_Tchaikovsky.mat');
for i=1:size(A,3)
    tchai(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Funk_EWF.mat');
for i=1:size(A,3)
    earth(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Funk_Funkadelic.mat');
for i=1:size(A,3)
    delic(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Funk_Sly.mat');
for i=1:size(A,3)
    stone(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Metal_Bodom.mat');
for i=1:size(A,3)
    bodom(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Metal_Necrophagist.mat');
for i=1:size(A,3)
    necro(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
A=importdata('Metal_Nile.mat');
for i=1:size(A,3)
    nile(:,i)=reshape(A(:,:,i),sp_w*sp_h,1);
end
clear A;

%% (test 0) Random Guess
% for any three choices, there's a 33% chance to get an answer right.

% solutions
test_sol= [zeros(1,18),1*ones(1,18),2*ones(1,18)]';
for i=1:perms
    rand_accuracy(i)=nnz((test_sol==randi([0 2],1,54)'))/length(test_sol);
end
% cross validation result statistics
u_accuracy=round(100*mean(rand_accuracy),1);
s_accuracy=round(100*std(rand_accuracy),2); 
result = strcat(num2str(u_accuracy)," +/- ",num2str(s_accuracy),"% (Accuracy to 1 Std. Dev.)");
plot(rand_accuracy), set(gca,'Ylim',[0,1]), legend(result,'Location','southeast')
title('Random Band Discrimination Performance')
ylabel(strcat("Accuracy: Given ",num2str(length(test_sol)*perms)," Trials"))
xlabel('Cross Validation Run Number')

%% (test 1) Band Classification:
%   From unused 5-second clips, determine if band is:
%   Beethoven, Children of Bodom, or Sly & the Family Stone
%   using full rank linear projection discrimination
%
data =[beeth bodom stone];
% decompose data
[u,s,v]=svd(data,'econ');
% grab band 'fingerprints', up to a given feature mode rank
vbeeth=v(1:48,1:rank);
vbodom=v(49:96,1:rank);
vstone=v(97:end,1:rank);

% cross validation of training/test
for i=1:perms
    % scramble data for train/test sets
    perm1=randperm(size(beeth,2));
    perm2=randperm(size(bodom,2));
    perm3=randperm(size(stone,2));
    % traning set
    train_set=[ vbeeth(perm1(1:30),:);...
                vbodom(perm2(1:30),:);...
                vstone(perm3(1:30),:)];
    % test set
    test_set= [ vbeeth(perm1(31:end),:);...
                vbodom(perm2(31:end),:);...
                vstone(perm3(31:end),:)];
    % ground truth (0=beethoven, 1=bodom, 2=sly & the family stone)
    sol_set = [zeros(1,30),1*ones(1,30),2*ones(1,30)]';
    test_sol= [zeros(1,18),1*ones(1,18),2*ones(1,18)]';
    % LDA discrimination tree
    [~,U,w,bm_line]=LDA_train(train_set(1:30,:)',train_set(31:60,:)',rank);
    test_SVD=U'*test_set';
    test_LDA=w'*test_SVD;
    beet_v_bodom=(test_LDA>bm_line); %0 for beethoven, 1 for bodom

    [~,U,w,bs_line]=LDA_train(train_set(1:30,:)',train_set(61:90,:)',rank);
    test_SVD=U'*test_set';
    test_LDA=w'*test_SVD;
    beet_v_sly=2*(test_LDA>bs_line); %0 for beethoven, 2 for sly & tfs

    [~,U,w,ms_line]=LDA_train(train_set(31:60,:)',train_set(61:90,:)',rank);
    test_SVD=U'*test_set';
    test_LDA=w'*test_SVD;
    bodom_v_sly=1+(test_LDA>ms_line); %1 for bodom, 2 for sly & tfs

    % results: each entry corresponds to a spectrogram column
    % 0=beethoven 1=bodom 2=sly & the family stone
    results=(mode([beet_v_bodom;beet_v_sly;bodom_v_sly]));
    results=(test_sol==results');
    accuracy(i)=nnz(results)/length(results);
end
% cross validation result statistics
u_accuracy=round(100*mean(accuracy),1);
s_accuracy=round(100*std(accuracy),2); 
result = strcat(num2str(u_accuracy)," +/- ",num2str(s_accuracy),"% (Accuracy to 1 Std. Dev.)");
plot(accuracy), set(gca,'Ylim',[0,1]), legend(result,'Location','southeast')
title('LDA Band Discrimination Performance')
ylabel(strcat("Accuracy: Given ",num2str(length(test_sol)*perms)," Trials"))
xlabel('Cross Validation Run Number')

%% (test 2) Classification Within Genre:
%   From unused 5-second clips, determine if band is:
%   Nile, Children of Bodom, or Necrophagist
%   using naive Bayesian discrimination

data =[bodom necro nile];
% decompose data
[u,s,v]=svd(data,'econ');
% grab band 'fingerprints', up to a given feature mode rank
vbodom=v(1:48,1:rank);
vnecro=v(49:96,1:rank);
vnile= v(97:end,1:rank);

% cross validation of training/test
for i=1:perms
    % scramble data for train/test sets
    perm1=randperm(size(bodom,2));
    perm2=randperm(size(necro,2));
    perm3=randperm(size(nile ,2));
    % traning set
    train_set=[ vbodom(perm1(1:30),:);...
                vnecro(perm2(1:30),:);...
                vnile(perm3(1:30),:)];
    % test set
    test_set= [ vbodom(perm1(31:end),:);...
                vnecro(perm2(31:end),:);...
                vnile(perm3(31:end),:)];
    % ground truth (1=bodom, 2=necro, 3=nile)
    sol_set = [ones(1,30),2*ones(1,30),3*ones(1,30)]';
    test_sol= [ones(1,18),2*ones(1,18),3*ones(1,18)]';
    % run naive bayes ppredictor
    nb=fitcnb(train_set,sol_set);
    predict=nb.predict(test_set);
    % use logic against solution set for accuracy
    result=(predict==test_sol); 
    accuracy(i)=nnz(result)/length(result);
end
% cross validation result statistics
u_accuracy=round(100*mean(accuracy),1);
s_accuracy=round(100*std(accuracy),2); 
result = strcat(num2str(u_accuracy)," +/- ",num2str(s_accuracy),...
    "% (Accuracy to 1 Std. Dev.)");
% cross validation visual
plot(accuracy)
set(gca,'Ylim',[0,1])
legend(result,'Location','southeast')
title('Naive Bayes Band Discrimination Performance')
ylabel(strcat("Accuracy: Given ",num2str(length(test_sol)*perms)," Trials"))
xlabel('Cross Validation Run Number')

%% (test 3) Genre Classification:
%  from 5-second clips of new bands, determine if genre is:
%   Classical, Metal, or Funk
%   using classification-and-regression-tree (CART) discrimination

data =[beeth mozrt tchai earth delic stone bodom necro nile];
% decompose data
[u,s,v]=svd(data,'econ'); clear data;
% grab genre 'fingerprints', up to a given feature mode rank
rank=5; %<----don't need many modes for decision tree to be accurate
vclas=v(1:144,1:rank);
vfunk=v(145:288,1:rank);
vmetl=v(289:end,1:rank);

% cross validation of training/test
for i=1:perms
    % scramble data for train/test sets
    perm1=randperm(size(vclas,1));
    perm2=randperm(size(vfunk,1));
    perm3=randperm(size(vmetl,1));
    % traning set
    train_set=[ vclas(perm1(1:90),:);...
                vfunk(perm2(1:90),:);...
                vmetl(perm3(1:90),:)];
    % test set
    test_set= [ vclas(perm1(91:end),:);...
                vfunk(perm2(91:end),:);...
                vmetl(perm3(91:end),:)];
    % ground truth (1=classic, 2=funk, 3=metal)
    sol_set = [ones(1,90),2*ones(1,90),3*ones(1,90)]';
    test_sol= [ones(1,54),2*ones(1,54),3*ones(1,54)]';
    % run naive bayes predictor
    tree=fitctree(train_set,sol_set);
    predict=tree.predict(test_set);
    % use logic against solution set for accuracy
    result=(predict==test_sol); 
    accuracy(i)=nnz(result)/length(result);
end
% cross validation result statistics
u_accuracy=round(100*mean(accuracy),1);
s_accuracy=round(100*std(accuracy),2); 
result = strcat(num2str(u_accuracy)," +/- ",num2str(s_accuracy),...
    "% (Accuracy to 1 Std. Dev.)");
% cross validation visual
plot(accuracy)
set(gca,'Ylim',[0,1])
legend(result,'Location','southeast')
title('Regression-Tree Genre Discrimination Performance')
ylabel(strcat("Accuracy: Given ",num2str(length(test_sol)*perms)," Trials"))
xlabel('Cross Validation Run Number')


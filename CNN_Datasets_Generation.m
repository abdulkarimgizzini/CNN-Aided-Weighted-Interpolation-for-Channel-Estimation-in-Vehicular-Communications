clc;clearvars;close all; warning('off','all');
% Load pre-defined DNN Testing Indices
load('./samples_indices_10.mat');
configuration = 'training'; % training or testing
% Define Simulation parameters
nDSC                      = 52;
nSym                      = 100;
mobility                  = 'VeryHigh';
ChType                    = 'VTV_SDWW';
WI_configuration          = '3P';
modu                      = 'QPSK';
scheme                    = 'WI_ALS';

if (isequal(configuration,'training'))
    indices = training_samples;
    EbN0dB           = 40; 
   
elseif(isequal(configuration,'testing'))
    indices = testing_samples;
    EbN0dB           = 0:5:40;    
end

Dataset_size     = size(indices,1);



SNR              = EbN0dB.';
N_SNR            = length(SNR);

for n_snr = 1:N_SNR

load(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_',configuration,'_simulation_' num2str(EbN0dB(n_snr)),'.mat'], 'True_Channels_Structure', [scheme '_Structure']);
scheme_Channels_Structure = eval([scheme '_Structure']);

Dataset_X        = zeros(nDSC*2, nSym, Dataset_size);
Dataset_Y        = zeros(nDSC*2, nSym, Dataset_size);


Dataset_X(1:nDSC,:,:)          = real(scheme_Channels_Structure);
Dataset_X(nDSC+1:2*nDSC,:,:)   = imag(scheme_Channels_Structure);
Dataset_Y(1:nDSC,:,:)          = real(True_Channels_Structure);
Dataset_Y(nDSC+1:2*nDSC,:,:)   = imag(True_Channels_Structure);

Dataset_X = permute(Dataset_X, [3, 1, 2 ]);
Dataset_Y = permute(Dataset_Y, [3, 1, 2 ]);

if (isequal(configuration,'training'))
    CNN_Datasets.('Train_X') =  Dataset_X;
    CNN_Datasets.('Train_Y') =  Dataset_Y;
elseif(isequal(configuration,'testing'))
    CNN_Datasets.('Test_X') =  Dataset_X;
    CNN_Datasets.('Test_Y') =  Dataset_Y;  
end
save(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_',scheme,'_CNN_',configuration,'_dataset_' num2str(EbN0dB(n_snr)),'.mat'],  'CNN_Datasets');

end
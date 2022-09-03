clc;clearvars;close all; warning('off','all');

mobility         = 'Low';
ChType          = 'VTV_UC';
WI_configuration = '1P';
modulation       = 'QPSK';
scheme           = 'WI_ALS';
CNN_Type         = 'SRCNN';
testing_samples  = 2;
load(['./',mobility,'_',ChType,'_',WI_configuration,'_',modulation,'_simulation_parameters']);

%% Simulation Parameters
SNR_p                     = (0:5:40)';
EbN0dB                    = SNR_p;
dpositions                = [1:6, 8:20, 22:31, 33:45, 47:52].';  
if(isequal(modulation,'QPSK'))
nBitPerSym  = 2;
elseif(isequal(modulation,'16QAM'))
nBitPerSym  = 4;
end
M                         = 2 ^ nBitPerSym;
Pow                       = mean(abs(qammod(0:(M-1),M)).^2); 
nSym                      = 100;
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
scramInit                 = 93;
nDSC                      = 52;
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
N_SNR                     = size(SNR_p,1);
Phf                       = zeros(N_SNR,1);
Err_scheme_CNN            = zeros(N_SNR,1);
Ber_scheme_CNN            = zeros(N_SNR,1);

for n_snr = 1:N_SNR
    tic;
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
     % Loading Simulation Parameters Results
    load(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_testing_simulation_',num2str(EbN0dB(n_snr)),'.mat']);
    % Loading scheme-DNN Results
    load(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_',scheme,'_',CNN_Type,'_Results_',num2str(EbN0dB(n_snr)),'.mat']);

     TestY = eval([scheme,'_CNN_test_y_',num2str(EbN0dB(n_snr))]);
     TestY = permute(TestY, [2,3,1]);
     TestY = TestY(1:nDSC,:,:) + 1i * TestY(nDSC+1:2*nDSC,:,:); 
     scheme_CNN = eval([scheme,'_CNN_corrected_y_',num2str(EbN0dB(n_snr))]);
     scheme_CNN = permute(scheme_CNN, [2,3,1]);
     scheme_CNN = scheme_CNN(1:nDSC,:,:) + 1i * scheme_CNN(nDSC+1:2*nDSC,:,:);
    for u = 1: size(scheme_CNN,3)

        H_scheme_CNN = scheme_CNN(:,:,u);

        Phf(n_snr) = Phf(n_snr) + mean(sum(abs(True_Channels_Structure(:,:,u)).^2)); 
        Err_scheme_CNN (n_snr) =  Err_scheme_CNN (n_snr) +  mean(sum(abs(H_scheme_CNN - True_Channels_Structure(:,:,u)).^2)); 
        
        % IEEE 802.11p Rx
        Bits_scheme_CNN     = de2bi((qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(: ,:,u) ./ H_scheme_CNN),M)));
        %Bits_AE_DNN     = de2bi((qamdemod(sqrt(Pow) * (EqualizedS(:,:,u) ),M)));
        Ber_scheme_CNN(n_snr)   = Ber_scheme_CNN(n_snr) + biterr(wlanScramble(vitdec((matintrlv((deintrlv(Bits_scheme_CNN(:),Random_permutation_Vector)).',Interleaver_Columns,16).'),poly2trellis(7,[171 133]),34,'trunc','hard'),93),TX_Bits_Stream_Structure(:,u));
     end
   toc;
end
Phf = Phf ./ testing_samples;
ERR_scheme_CNN = Err_scheme_CNN ./ (testing_samples * Phf); 
BER_scheme_CNN = Ber_scheme_CNN/ (testing_samples * nSym * nDSC * nBitPerSym);


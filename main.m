clc;clearvars;close all; warning('off','all');
ch_func = Channel_functions();
%% --------OFDM Parameters - Given in IEEE 802.11p Spec--
ofdmBW                 = 10 * 10^6 ;     % OFDM bandwidth (Hz)
nFFT                   = 64;             % FFT size 
nDSC                   = 52;             % Number of data subcarriers
nPSC                   = 0;              % Number of pilot subcarriers
nZSC                   = 12;             % Number of zeros subcarriers
nUSC                   = nDSC + nPSC;    % Number of total used subcarriers
K                      = nUSC + nZSC;    % Number of total subcarriers
nSym                   = 100;             % Number of OFDM symbols within one frame
deltaF                 = ofdmBW/nFFT;    % Bandwidth for each subcarrier - include all used and unused subcarriers 
Tfft                   = 1/deltaF;       % IFFT or FFT period = 6.4us
Tgi                    = Tfft/4;         % Guard interval duration - duration of cyclic prefix - 1/4th portion of OFDM symbols = 1.6us
Tsignal                = Tgi+Tfft;       % Total duration of BPSK-OFDM symbol = Guard time + FFT period = 8us
K_cp                   = nFFT*Tgi/Tfft;  % Number of symbols allocated to cyclic prefix 
% Pre-defined preamble in frequency domain
dp = [ 0  0 0 0 0 0 +1 +1 -1 -1 +1  +1 -1  +1 -1 +1 +1 +1 +1 +1 +1 -1 -1 +1 +1 -1 +1 -1 +1 +1 +1 +1 0 +1 -1 -1 +1 +1 -1 +1 -1 +1 -1 -1 -1 -1 -1 +1 +1 -1 -1 +1 -1 +1 -1 +1 +1 +1 +1 0 0 0 0 0];
Ep                     = 1;              % pramble power per sample
dp                     = fftshift(dp);   % Shift zero-frequency component to center of spectrum    
predefined_preamble    = dp;
Kset                   = find(dp~=0);               % set of allocated subcarriers                  
Kon                    = length(Kset);              % Number of active subcarriers
dp                     = sqrt(Ep)*dp.';
xp                     = sqrt(K)*ifft(dp);
xp_cp                  = [xp(end-K_cp+1:end); xp];  
preamble_80211p        = repmat(xp_cp,1,2);  
p1                     = xp_cp;         
used_locations         = [2:27, 39:64].';
pilots_locations       = [8,22,44,58].'; % Pilot subcarriers positions
pilots                 = [1 1 1 -1].';
data_locations         = [2:7, 9:21, 23:27, 39:43, 45:57, 59:64].'; % Data subcarriers positions
ppositions             = [7,21, 32,46].';                           % Pilots positions in Kset
dpositions             = [1:6, 8:20, 22:31, 33:45, 47:52].';        % Data positions in Kset
pilots_locations_Full  = [2,6,12,18,24,27,39,45,51,57,61,64].'; % Pilot subcarriers positions
%% ------ Bits Modulation Technique------------------------------------------
modu                      = 'QPSK';
Mod_Type                  = 1;              % 0 for BPSK and 1 for QAM 
if(Mod_Type == 0)
    nBitPerSym            = 1;
    Pow                   = 1;
    %BPSK Modulation Objects
    bpskModulator         = comm.BPSKModulator;
    bpskDemodulator       = comm.BPSKDemodulator;
    M                     = 1;
elseif(Mod_Type == 1)
    if(strcmp(modu,'QPSK') == 1)
         nBitPerSym       = 2; 
    elseif (strcmp(modu,'16QAM') == 1)
         nBitPerSym       = 4; 
    elseif (strcmp(modu,'64QAM') == 1)
         nBitPerSym       = 6; 
    end
    M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
    Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM    
    Constellation         =  1/sqrt(Pow) * qammod(0:(M-1),M); % 
end

%% ---------Scrambler Parameters---------------------------------------------
scramInit                 = 93; % As specidied in IEEE 802.11p Standard [1011101] in binary representation
%% ---------Convolutional Coder Parameters-----------------------------------
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
rate                      = 1/2;
%% -------Interleaver Parameters---------------------------------------------
% Matrix Interleaver
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
% General Block Interleaver
Random_permutation_Vector = randperm(nBitPerSym*nDSC*nSym); % Permutation vector
%% -----------------Vehicular Channel Model Parameters--------------------------
mobility                  = 'VeryHigh';
ChType                    = 'VTV_SDWW';             % Channel model
ch_l                      = [6 7 8 9];    
fs                        = K*deltaF;               % Sampling frequency in Hz, here case of 802.11p with 64 subcarriers and 156250 Hz subcarrier spacing
fc                        = 5.9e9;                  % Carrier Frequecy in Hz.
vel                       = 48;                    % Moving speed of user in km
c                         = 3e8;                    % Speed of Light in m/s
fD                        = 250;%(vel/3.6)/c*fc;         % Doppler freq in Hz
rchan                     = ch_func.GenFadingChannel(ChType, fD, fs);
%% Simulation Parameters 
load('./samples_indices_1000.mat');
configuration     = 'training'; % training or testing
WI_configuration  = '1P';
if (isequal(configuration,'training'))
    indices = training_samples;
    EbN0dB           = 40; 
elseif(isequal(configuration,'testing'))
    indices = testing_samples;
    EbN0dB           = 0:5:40;         
end

%% ---------Bit to Noise Ratio------------------%
SNR_p                     = EbN0dB + 10*log10(K/nDSC) + 10*log10(K/(K + K_cp)) + 10*log10(nBitPerSym) + 10*log10(rate);
SNR_p                     = SNR_p.';
N0                        = Ep*10.^(-SNR_p/10);
N_CH                      = size(indices,1); 
N_SNR                     = length(SNR_p); 

%% DFT Interpolation
D                       = dftmtx (nFFT);
Dt_LS                   = D(Kset,ch_l+1);
DpLS                    = D(Kset,ch_l+1);
temp_LS = ((DpLS' * DpLS)^-1) * DpLS';
H_Interpolation_LS   = Dt_LS * temp_LS; 

% Full DFT
Dt_Full                   = D(Kset,ch_l+1);
Dp_Full                   = D(pilots_locations_Full,ch_l+1);
temp_Full                 = ((Dp_Full' * Dp_Full)^-1) * Dp_Full';
H_Interpolation_Full      = Dt_Full * temp_Full;
ERR_DFT                   = trace (H_Interpolation_Full * H_Interpolation_Full');



% Normalized mean square error (NMSE) vectors
Err_ALS    = zeros(N_SNR,1);
Err_SLS    = zeros(N_SNR,1);
Err_LP     = zeros(N_SNR,1);

% Bit error rate (BER) vectors
Ber_Ideal  = zeros(N_SNR,1);
Ber_ALS    = zeros(N_SNR,1);
Ber_SLS    = zeros(N_SNR,1);
Ber_LP     = zeros(N_SNR,1);
% average channel power E(|hf|^2)
Phf_H_Total                  = zeros(N_SNR,1);

%% Simulation Loop
for n_snr = 1:N_SNR
    disp(['Running Simulation, SNR = ', num2str(EbN0dB(n_snr))]);
    tic;      
    
     TX_Bits_Stream_Structure                 = zeros(nDSC * nSym  * nBitPerSym *rate, N_CH);
     Received_Symbols_FFT_Structure           = zeros(Kon,nSym, N_CH);
     True_Channels_Structure                  = zeros(Kon, nSym, N_CH);
     WI_ALS_Structure                         = zeros(Kon, nSym, N_CH);
     WI_SLS_Structure                         = zeros(Kon, nSym, N_CH);
     WI_LP_Structure                          = zeros(Kon, nSym, N_CH);
         
    for n_ch = 1:N_CH % loop over channel realizations
        % Bits Stream Generation 
        Bits_Stream_Coded = randi(2, nDSC * nSym  * nBitPerSym * rate,1)-1;
        % Data Scrambler 
        scrambledData = wlanScramble(Bits_Stream_Coded,scramInit);
        % Convolutional Encoder
        dataEnc = convenc(scrambledData,trellis);
        % Interleaving
        % Matrix Interleaving
        codedata = dataEnc.';
        Matrix_Interleaved_Data = matintrlv(codedata,Interleaver_Rows,Interleaver_Columns).';
        % General Block Interleaving
        General_Block_Interleaved_Data = intrlv(Matrix_Interleaved_Data,Random_permutation_Vector);
        % Bits Mapping: M-QAM Modulation
        TxBits_Coded = reshape(General_Block_Interleaved_Data,nDSC , nSym  , nBitPerSym);
        % Gray coding goes here
        TxData_Coded = zeros(nDSC ,nSym);
        for m = 1 : nBitPerSym
           TxData_Coded = TxData_Coded + TxBits_Coded(:,:,m)*2^(m-1);
        end
        % M-QAM Modulation
         Modulated_Bits_Coded  =1/sqrt(Pow) * qammod(TxData_Coded,M);
        
         % OFDM Frame Generation
         OFDM_Frame_Coded = zeros(K,nSym);
         OFDM_Frame_Coded(used_locations,:) = Modulated_Bits_Coded;
         % Taking FFT, the term (nFFT/sqrt(nDSC)) is for normalizing the power of transmit symbol to 1 
         IFFT_Data_Coded =  sqrt(K)*ifft(OFDM_Frame_Coded); 
         % Appending cylic prefix
         CP_Coded = IFFT_Data_Coded((K - K_cp +1):K,:);
         IFFT_Data_CP_Coded = [CP_Coded; IFFT_Data_Coded];
         % Appending preambles symbol accoding to the WI_configuration
         if(isequal(WI_configuration,'1P'))
            
              
         IFFT_Data_CP_Preamble_Coded = [preamble_80211p IFFT_Data_CP_Coded p1];
         % ideal estimation
         release(rchan);
         rchan.Seed = indices(n_ch,1);
         [ h, y ] = ch_func.ApplyChannel(rchan, IFFT_Data_CP_Preamble_Coded, K_cp);
        
        
        yp = y((K_cp+1):end,1:2);
        yp1 = y((K_cp+1):end,end);
        y  = y((K_cp+1):end,3:end-1);
        
        yFD =  sqrt(1/K)*fft(y);
        yfp =  sqrt(1/K)*fft(yp); 
        yfp1 =  sqrt(1/K)*fft(yp1); 
      
        
        h = h((K_cp+1):end,:);
        hf = fft(h);    
        hf  = hf(:,3:end-1);

        
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + mean(sum(abs(hf(Kset,:)).^2));
        %add noise
        noise_p = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);
        noise_p1 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        yfp_r = yfp +  noise_p;
        yfp1_r = yfp1 +  noise_p1;
        noise_OFDM_Symbols = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);
        y_r   = yFD + noise_OFDM_Symbols;    
       %% Channel Estimation
       % ALS
       he_ALS_P1 = H_Interpolation_LS * ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       he_ALS_P2 = H_Interpolation_LS * (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
       P_ALS = [he_ALS_P1 he_ALS_P2].';
       
       % SLS
       he_LS_P1 = ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       he_LS_P2 = (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
       P_SLS = [he_LS_P1 he_LS_P2].';
       
       % LP
       HLS_P2_Full = yfp1_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
       H_DFT_Interpolation_ZF_P2_Full = H_Interpolation_Full * HLS_P2_Full; 
       P_LP = [he_LS_P1 H_DFT_Interpolation_ZF_P2_Full(:,1)].';
        
        
        % Interpolation Matrix Calculation
        H_WI_ALS = zeros(nDSC,nSym);
        H_WI_SLS = zeros(nDSC,nSym);
        H_WI_LP  = zeros(nDSC,nSym);
        noise_power_OFDM_Symbols = var(noise_OFDM_Symbols);
        [Ck_Cofficients] = Interpolation_Cofficients_FP(nSym, fD, Tsignal, noise_power_OFDM_Symbols);   
        [Ck_Cofficients_LP] = Interpolation_Cofficients_LP(nSym, fD, Tsignal, noise_power_OFDM_Symbols, ERR_DFT);
      
        for li = 1:nSym 
            H_WI_ALS(:,li) = Ck_Cofficients(li,:) * P_ALS; 
            H_WI_SLS(:,li) = Ck_Cofficients(li,:) * P_SLS;
            H_WI_LP(:,li)  = Ck_Cofficients_LP(li,:) * P_LP;
        end
        
                 
            
       Err_ALS (n_snr) = Err_ALS (n_snr) + mean(sum(abs(H_WI_ALS - hf(Kset,:)).^2));
       Err_SLS (n_snr) = Err_SLS (n_snr) + mean(sum(abs(H_WI_SLS - hf(Kset,:)).^2));
       Err_LP (n_snr)  = Err_LP (n_snr) + mean(sum(abs(H_WI_LP - hf(Kset,:)).^2));
      
    
       %%    IEEE 802.11p Rx     
        Bits_Ideal          = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ hf(Kset,:)),M));
        Bits_ALS            = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_ALS),M));
        Bits_SLS            = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_SLS),M));
        Bits_LP             = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_LP),M));
      
       Ber_Ideal (n_snr)    = Ber_Ideal (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_Ideal(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_ALS (n_snr)      = Ber_ALS (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_ALS(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_SLS (n_snr)      = Ber_SLS (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_SLS(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_LP (n_snr)       = Ber_LP (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_LP(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
                        
         elseif(isequal(WI_configuration,'2P'))
             
             
         First_Frame_Half = IFFT_Data_CP_Coded(:,1 : nSym/2);
         Second_Frame_Half = IFFT_Data_CP_Coded(:, (nSym/2) + 1 : end);
         IFFT_Data_CP_Preamble_Coded = [preamble_80211p First_Frame_Half p1 Second_Frame_Half p1];

        % ideal estimation
        release(rchan);
        rchan.Seed = indices(n_ch,1);
        [ h, y ] = ch_func.ApplyChannel( rchan, IFFT_Data_CP_Preamble_Coded, K_cp);

        yp = y((K_cp+1):end,1:2);
        yp1 = y((K_cp+1):end, (nSym/2) +3);
        yp2 = y((K_cp+1):end, end);
        
        
        
        y  = [y((K_cp+1):end,3: (nSym/2) + 2) y((K_cp+1):end, (nSym/2) + 4 : end-1)];
        
        
        yFD = sqrt(1/K)*fft(y);
        yfp = sqrt(1/K)*fft(yp); 
        yfp1 = sqrt(1/K)*fft(yp1); 
        yfp2 = sqrt(1/K)*fft(yp2);
      
        
        h = h((K_cp+1):end,:);
        hf = fft(h);
      
       
        hf  = [hf(:,3:(nSym/2) + 2) hf(:, (nSym/2) + 4 : end-1)];

      
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr)  + mean(sum(abs(hf(Kset,:)).^2));
        %add noise
        noise_p = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);
        noise_p1 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        noise_p2 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        yfp_r = yfp +  noise_p;
        yfp1_r = yfp1 +  noise_p1;
        yfp2_r = yfp2 +  noise_p2;
        noise_OFDM_Symbols = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);
        y_r   = yFD + noise_OFDM_Symbols;
       
     
       %% Channel Estimation
       % ALS        
       Ahe_LS_P1 =  H_Interpolation_LS * ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       Ahe_LS_P2 = H_Interpolation_LS * (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
       Ahe_LS_P3 = H_Interpolation_LS * (yfp2_r(Kset,1)./ predefined_preamble(Kset).');
       AP1 = [Ahe_LS_P1 Ahe_LS_P2].';
       AP2 = [Ahe_LS_P2 Ahe_LS_P3].';
        
        % SLS
        he_LS_P1 =  ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
        he_LS_P2 = (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
        he_LS_P3 = (yfp2_r(Kset,1)./ predefined_preamble(Kset).');
        P1_SLS = [he_LS_P1 he_LS_P2].';
        P2_SLS = [he_LS_P2 he_LS_P3].';
       
        % LP
        HLS_P2_Full = yfp1_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
        HLS_P3_Full = yfp2_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
        H_DFT_Interpolation_ZF_P2_Full = H_Interpolation_Full * HLS_P2_Full;
        H_DFT_Interpolation_ZF_P3_Full = H_Interpolation_Full * HLS_P3_Full; 
        P1_LP = [he_LS_P1 H_DFT_Interpolation_ZF_P2_Full(:,1)].';
        P2_LP = [H_DFT_Interpolation_ZF_P2_Full(:,1) H_DFT_Interpolation_ZF_P3_Full(:,1)].';
        
        % Interpolation Matrix Calculation
        H_WI_ALS = zeros(nDSC,nSym);
        H_WI_SLS = zeros(nDSC,nSym);
        H_WI_LP  = zeros(nDSC,nSym);
        noise_power_OFDM_Symbols = var(noise_OFDM_Symbols);
        [Ck_Cofficients] = Interpolation_Cofficients_FP(nSym/2, fD, Tsignal, noise_power_OFDM_Symbols);
        [Ck_Cofficients_LP] = Interpolation_Cofficients_LP(nSym/2, fD, Tsignal, noise_power_OFDM_Symbols, ERR_DFT); 
        Ck_Cofficients = [Ck_Cofficients; Ck_Cofficients];
        Ck_Cofficients_LP = [Ck_Cofficients_LP;Ck_Cofficients_LP];
       
        for li = 1:nSym/2 

            H_WI_ALS(:,li) = Ck_Cofficients(li,:) * AP1; 
            H_WI_SLS(:,li) = Ck_Cofficients(li,:) * P1_SLS;
            H_WI_LP(:,li)  = Ck_Cofficients_LP(li,:) * P1_LP;
        end
       
        for lo = nSym/2 + 1 : nSym 
         
            H_WI_ALS(:,lo) = Ck_Cofficients(lo,:) * AP2; 
            H_WI_SLS(:,lo) = Ck_Cofficients(lo,:) * P2_SLS;
            H_WI_LP(:,lo)  = Ck_Cofficients_LP(lo,:) * P2_LP;
        
        end
        
                 
            
       Err_ALS (n_snr) = Err_ALS (n_snr) + mean(sum(abs(H_WI_ALS - hf(Kset,:)).^2));
       Err_SLS (n_snr) = Err_SLS (n_snr) + mean(sum(abs(H_WI_SLS - hf(Kset,:)).^2));
       Err_LP (n_snr)  = Err_LP (n_snr) + mean(sum(abs(H_WI_LP - hf(Kset,:)).^2));
      
    
       %%    IEEE 802.11p Rx     
        Bits_Ideal          = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ hf(Kset,:)),M));
        Bits_ALS            = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_ALS),M));
        Bits_SLS            = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_SLS),M));
        Bits_LP             = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_LP),M));
      
       Ber_Ideal (n_snr)    = Ber_Ideal (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_Ideal(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_ALS (n_snr)      = Ber_ALS (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_ALS(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_SLS (n_snr)      = Ber_SLS (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_SLS(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_LP (n_snr)       = Ber_LP (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_LP(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       
         
         elseif(isequal(WI_configuration,'3P'))
             
              First_Frame_Half = IFFT_Data_CP_Coded(:,1 : 33);
         Second_Frame_Half = IFFT_Data_CP_Coded(:, 34 : 66);
         Third_Frame_Half = IFFT_Data_CP_Coded(:, 67 : 100);
         IFFT_Data_CP_Preamble_Coded = [preamble_80211p First_Frame_Half p1 Second_Frame_Half p1 Third_Frame_Half p1];
        
        % ideal estimation
        release(rchan);
        rchan.Seed = indices(n_ch,1);
        [ h, y ] = ch_func.ApplyChannel( rchan, IFFT_Data_CP_Preamble_Coded, K_cp);


        yp = y((K_cp+1):end,1:2);
        yp1 = y((K_cp+1):end, 36);
        yp2 = y((K_cp+1):end, 70);
        yp3 = y((K_cp+1):end, 105);
        
        
        
        y  = [y((K_cp+1):end,3: 35) y((K_cp+1):end, 37:69) y((K_cp+1):end, 71: 104)];
   
        yFD = sqrt(1/K)*fft(y);
        yfp = sqrt(1/K)*fft(yp); 
        yfp1 = sqrt(1/K)*fft(yp1); 
        yfp2 = sqrt(1/K)*fft(yp2);
        yfp3 = sqrt(1/K)*fft(yp3);

      
        
        h = h((K_cp+1):end,:);
        hf = fft(h);       
        hf_true_preambles = hf(Kset,1) + hf(Kset,2) ./2; 
        hf_complete = [hf_true_preambles hf(Kset,3:end)];
        hf  = [hf(:,3: 35) hf(:, 37:69) hf(:, 71:104)];

      
        Phf_H_Total(n_snr) = Phf_H_Total(n_snr) + + mean(sum(abs(hf(Kset,:)).^2));
        %add noise
        noise_p = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,2], 1);
        noise_p1 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        noise_p2 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        noise_p3 = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,1], 1);
        yfp_r = yfp +  noise_p;
        yfp1_r = yfp1 +  noise_p1;
        yfp2_r = yfp2 +  noise_p2;
        yfp3_r = yfp3 +  noise_p3;
        noise_OFDM_Symbols = sqrt(N0(n_snr))*ch_func.GenRandomNoise([K,size(yFD,2)], 1);
        y_r   = yFD + noise_OFDM_Symbols;
       
      
       %% Channel Estimation
       % ALS
       he_LS_P1 = H_Interpolation_LS * ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
       he_LS_P2 = H_Interpolation_LS * (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
       he_LS_P3 = H_Interpolation_LS * (yfp2_r(Kset,1)./ predefined_preamble(Kset).');
       he_LS_P4 = H_Interpolation_LS * (yfp3_r(Kset,1)./ predefined_preamble(Kset).');
       AP1 = [he_LS_P1 he_LS_P2].';
       AP2 = [he_LS_P2 he_LS_P3].';
       AP3 = [he_LS_P3 he_LS_P4].';
       
               % SLS
        he_SLS_P1 =  ((yfp_r(Kset,1) + yfp_r(Kset,2))./(2.*predefined_preamble(Kset).'));
        he_SLS_P2 = (yfp1_r(Kset,1)./ predefined_preamble(Kset).');
        he_SLS_P3 = (yfp2_r(Kset,1)./ predefined_preamble(Kset).');
        he_SLS_P4 = (yfp3_r(Kset,1)./ predefined_preamble(Kset).');
        P1_SLS = [he_SLS_P1 he_SLS_P2].';
        P2_SLS = [he_SLS_P2 he_SLS_P3].';
        P3_SLS = [he_SLS_P3 he_SLS_P4].';
       
        % LP
        HLS_P2_Full = yfp1_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
        HLS_P3_Full = yfp2_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
        HLS_P4_Full = yfp3_r(pilots_locations_Full,1)./ predefined_preamble(pilots_locations_Full).';
        H_DFT_Interpolation_ZF_P2_Full = H_Interpolation_Full * HLS_P2_Full;
        H_DFT_Interpolation_ZF_P3_Full = H_Interpolation_Full * HLS_P3_Full; 
        H_DFT_Interpolation_ZF_P4_Full = H_Interpolation_Full * HLS_P4_Full;        
        P1_LP = [he_LS_P1 H_DFT_Interpolation_ZF_P2_Full(:,1)].';
        P2_LP = [H_DFT_Interpolation_ZF_P2_Full(:,1) H_DFT_Interpolation_ZF_P3_Full(:,1)].';
        P3_LP = [H_DFT_Interpolation_ZF_P3_Full(:,1) H_DFT_Interpolation_ZF_P4_Full(:,1)].';
        
        % Interpolation Matrix Calculation
        H_WI_ALS = zeros(nDSC,nSym);
        H_WI_SLS = zeros(nDSC,nSym);
        H_WI_LP  = zeros(nDSC,nSym);

        noise_power_OFDM_Symbols = var(noise_OFDM_Symbols);
        [Ck_Cofficients_FP] = Interpolation_Cofficients_FP(33, fD, Tsignal, noise_power_OFDM_Symbols);
        [Ck_Cofficients_FP1] = Interpolation_Cofficients_FP(34, fD, Tsignal, noise_power_OFDM_Symbols);       
        [Ck_Cofficients_LP] = Interpolation_Cofficients_LP(33, fD, Tsignal, noise_power_OFDM_Symbols, ERR_DFT); 
        [Ck_Cofficients_LP1] = Interpolation_Cofficients_LP(34, fD, Tsignal, noise_power_OFDM_Symbols, ERR_DFT); 
       
        Ck_Cofficients = [Ck_Cofficients_FP; Ck_Cofficients_FP;Ck_Cofficients_FP1];
        Ck_Cofficients_LP = [Ck_Cofficients_LP; Ck_Cofficients_LP;Ck_Cofficients_LP1];

       
        for li = 1:33 
           
        
            H_WI_ALS(:,li) = Ck_Cofficients(li,:) * AP1; 
            H_WI_SLS(:,li) = Ck_Cofficients(li,:) * P1_SLS;
            H_WI_LP(:,li)  = Ck_Cofficients_LP(li,:) * P1_LP;
        
        end
       
        for lo = 34 : 66 
            
        
            H_WI_ALS(:,lo) = Ck_Cofficients(lo,:) * AP2; 
            H_WI_SLS(:,lo) = Ck_Cofficients(lo,:) * P2_SLS;
            H_WI_LP(:,lo)  = Ck_Cofficients_LP(lo,:) * P2_LP;
        
        end
        
        for ly = 67 : 100 
          
            H_WI_ALS(:,ly) = Ck_Cofficients(ly,:) * AP3; 
            H_WI_SLS(:,ly) = Ck_Cofficients(ly,:) * P3_SLS;
            H_WI_LP(:,ly)  = Ck_Cofficients_LP(ly,:) * P3_LP;
        end
        
                 
            
       Err_ALS (n_snr) = Err_ALS (n_snr) + mean(sum(abs(H_WI_ALS - hf(Kset,:)).^2));
       Err_SLS (n_snr) = Err_SLS (n_snr) + mean(sum(abs(H_WI_SLS - hf(Kset,:)).^2));
       Err_LP (n_snr)  = Err_LP (n_snr) + mean(sum(abs(H_WI_LP - hf(Kset,:)).^2));
      
    
       %%    IEEE 802.11p Rx     
        Bits_Ideal          = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ hf(Kset,:)),M));
        Bits_ALS            = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_ALS),M));
        Bits_SLS            = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_SLS),M));
        Bits_LP             = de2bi(qamdemod(sqrt(Pow) * (y_r(Kset ,:) ./ H_WI_LP),M));
      
       Ber_Ideal (n_snr)    = Ber_Ideal (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_Ideal(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_ALS (n_snr)      = Ber_ALS (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_ALS(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_SLS (n_snr)      = Ber_SLS (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_SLS(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
       Ber_LP (n_snr)       = Ber_LP (n_snr) + biterr(wlanScramble((vitdec((matintrlv((deintrlv(Bits_LP(:),Random_permutation_Vector)).',Interleaver_Columns,Interleaver_Rows).'),trellis,tbl,'trunc','hard')),scramInit),Bits_Stream_Coded);
         
         end
      
           TX_Bits_Stream_Structure(:, n_ch) = Bits_Stream_Coded;
           Received_Symbols_FFT_Structure(:,:,n_ch) = y_r(Kset ,:);
           True_Channels_Structure(:,:,n_ch) = hf(Kset,:);
           WI_ALS_Structure(:,:,n_ch)  =  H_WI_ALS; 
           WI_SLS_Structure(:,:,n_ch)  =  H_WI_SLS; 
           WI_LP_Structure(:,:,n_ch)   =  H_WI_LP; 
        

              
    end 
   
    if (isequal(configuration,'training'))

          save(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_training_simulation_' num2str(EbN0dB(n_snr))],...              
          'True_Channels_Structure',...
          'WI_ALS_Structure',...
          'WI_SLS_Structure',...
          'WI_LP_Structure') 
       
    elseif(isequal(configuration,'testing'))
          
          save(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_testing_simulation_' num2str(EbN0dB(n_snr))],...              
          'TX_Bits_Stream_Structure',...
          'Received_Symbols_FFT_Structure',...
          'True_Channels_Structure',...
          'WI_ALS_Structure',...
          'WI_SLS_Structure',...
          'WI_LP_Structure')        
    end
    toc;
end 

%% Bit Error Rate (BER)
BER_Ideal  = Ber_Ideal /(N_CH * nSym * nDSC * nBitPerSym);
BER_ALS    = Ber_ALS  / (N_CH * nSym * nDSC * nBitPerSym);
BER_SLS    = Ber_SLS  / (N_CH * nSym * nDSC * nBitPerSym);
BER_LP     = Ber_LP  / (N_CH * nSym * nDSC * nBitPerSym);
%% Normalized Mean Square Error
Phf_H      = Phf_H_Total/(N_CH);
ERR_ALS    = Err_ALS / (Phf_H * N_CH); 
ERR_SLS    = Err_SLS / (Phf_H * N_CH); 
ERR_LP     = Err_LP / (Phf_H * N_CH); 
%% Plotting Results
figure,
p1 = semilogy(EbN0dB, BER_Ideal,'k-o','LineWidth',2);
hold on;
p2 = semilogy(EbN0dB, BER_ALS,'r--o','LineWidth',2);
hold on;
p3 = semilogy(EbN0dB, BER_SLS,'b--o','LineWidth',2);
hold on;
p4 = semilogy(EbN0dB, BER_LP,'g--o','LineWidth',2);
hold on;
grid on;
legend([p1(1),p2(1),p3(1),p4(1)],{'Perfect Channel','WI-ALS','WI-SLS','WI-LP'});
xlabel('SNR(dB)');
ylabel('BER');

figure,
p1 = semilogy(EbN0dB, ERR_ALS,'r--o','LineWidth',2);
hold on;
p2 = semilogy(EbN0dB, ERR_SLS,'b--o','LineWidth',2);
hold on;
p3 = semilogy(EbN0dB, ERR_LP,'g--o','LineWidth',2);
hold on;
grid on;
legend([p1(1),p2(1),p3(1)],{'WI-ALS','WI-SLS','WI-LP'});
xlabel('SNR(dB)');
ylabel('NMSE');

if (isequal(configuration,'training'))
       
elseif(isequal(configuration,'testing'))
      save(['./',mobility,'_',ChType,'_',WI_configuration,'_',modu,'_simulation_parameters'],...
      'BER_Ideal','BER_ALS','BER_SLS','BER_LP',...
      'ERR_ALS','ERR_SLS','ERR_LP',...
       'predefined_preamble','modu','Kset','Random_permutation_Vector','fD','ChType');
end




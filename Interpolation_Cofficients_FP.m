function [Ck_Cofficients] = Interpolation_Cofficients1(nSym, fD, Tsymbol, noise_power_OFDM_Symbols)
Ck_Cofficients = zeros(nSym,2);
for i = 2:nSym +1
A1_Temp = besselj(2*pi*fD*(i - 1)* Tsymbol, 1);
A2_Temp = besselj(2*pi*fD*(nSym + 1 - i)* Tsymbol, 1);
A_Temp = [A1_Temp A2_Temp];
B14_Temp = 1 + ((noise_power_OFDM_Symbols(1, i - 1)) / 2);
B44_Temp = 1 + ((noise_power_OFDM_Symbols(1, i - 1)));
B23_Temp = besselj(2*pi*fD*nSym* Tsymbol, 1);  
B_Temp = [B14_Temp B23_Temp ; B23_Temp B44_Temp];
Ck_Cofficients(i - 1,:) = A_Temp /(B_Temp);
end
end


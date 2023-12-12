This repository includes the source code of the CNN-based channel estimators proposed in "CNN Aided Weighted Interpolation for Channel Estimation in Vehicular Communications" paper [1] that is published in the  IEEE Transactions on Vehicular Technology, 2021. Please note that the Tx-Rx OFDM processing is implemented in Matlab and the LSTM processing is implemented in python (Keras).


### Files Description 
- Main.m: The main simulation file, where the simulation parameters (Channel model, OFDM parameters, Modulation scheme, etc...) are defined. 
- Channel_functions.m: Includes the pre-defined vehicular channel models [3] for different mobility conditions.
- CNN_Datasets_Generation.m: Generating the LSTM training/testing datasets.
- CNN_Results_Processing.m: Processing the testing results genertead by the LSTM testing and caculate the BER and NMSE results of the CNN-based estimator.
- CNN.py: The CNN (SRCNN & DNCNN) training/testing is performed employing the generated training/testing datasets. The file should be executed twice as follows:
	- **Step1: Training by executing this command python CNN.py  Mobility Channel_Model WI_Configuration Modulation_Order Channel_Estimator Training_SNR CNN_Type CNN_Input Epochs Batch_size**
	- **Step2: Testing by executing this command: python CNN.py  Mobility Channel_Model WI_Configuration Modulation_Scheme Channel_Estimator Testing_SNR CNN_Type CNN_Input** 
> ex: python CNN.py  Low VTV_UC 1P QPSK WI_ALS 40 SRCNN 104 500 128

> ex: python CNN.py Low VTV_UC 1P QPSK WI_ALS 40 SRCNN 104
		
### Running Steps:
1. Run the IDX_Generation.m in order to genertae the dataset indices, training dataset size, and testing dataset size.
2. Run the main.m file two times as follows:
	- Specify all the simulation parameters like: the number of OFDM symbols, channel model, mobility scenario, modulatio order, SNR range, etc.
	- Specify the path of the generated indices in step (1).
	- The first time for generating the traininig simulation file (set the configuration = 'training' in the code).
	- The second time for generating the testing simulations files (set the configuration = 'testing' in the code).
	- After that, the generated simulations files will be saved in your working directory.
3. Run the CNN_Datasets_Generation.m also two times by changing the configuration as done in step (2) in addition to specifying the channel estimation scheme as well as the OFDM simulation parameters. This step generates the LSTM training/testing datasets.
4. Run the CNN.py file also two times in order to perform the training first then the testing as mentioned in the LSTM.py file description.
5. After finishing step 4, the CNN results will be saved as a .mat files. Then you need to run the CNN_Results_Processing.m file in order to get the NMSE and BER results of the studied channel estimation scheme.

### References
- [1] A. Karim Gizzini, M. Chafii, A. Nimr, R. M. Shubair and G. Fettweis, "CNN Aided Weighted Interpolation for Channel Estimation in Vehicular Communications," in IEEE Transactions on Vehicular Technology, vol. 70, no. 12, pp. 12796-12811, Dec. 2021, doi: 10.1109/TVT.2021.3120267.

For more information and questions, please contact me on abdulkarimgizzini@gmail.com 

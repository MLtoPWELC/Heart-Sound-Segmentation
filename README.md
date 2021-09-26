# Heart-Sound-Segmentation
paper:Temporal Convolutional Network Connected with an Anti-Arrhythmia Hidden Semi-Markov Model for Heart Sound Segmentation

paper-website:https://www.mdpi.com/2076-3417/10/20/7049

The current code is not the final version. It can be used as a reference at present.

remarks:
download source of PN-training-a：
name ：training-a set
https://physionet.org/content/challenge-2016/1.0.0/training-a/#files-panel

download source of LH-training：
name：example_data.mat
https://physionet.org/content/hss/1.0/

1. The operation of training-a set because of lack of ECG or sever noise:
delete: No.41 No.117 No.137 No.220 No.233 No.314 (6 in total)
truncation: No.47 (1:14539 retain) No.105 (1:25951 retain)

2. LH_training_label.npy and PN_training_a_label.npy are the ground truth label of corresponding data, and they are down sampled to 50Hz because the original label files are too big. You can use the code ‘expand_qt.m’ described later to restore to original length.

3. TCN_model.py is only an example code which can not be directly run. It can be the reference for reproduction code.

4. The example of Viterbi code is in folder “viterbi” which can not directly run. “SEGMENTOFSOUND.m” is the main function.   “viterbiDecodePCG_Springer.m” is the core function of Viterbi algorithm, and the improved methods is implemented in this function. ” F1_score.m” is the function that gets the F1 scores. “expand_qt.m” is the function to expand the signals or annotations of 50 Hz to 1000 Hz. The functions in this file are all derived from or based on the open source code developed by David Springer for comparison purposes in the paper D. Springer et al., "Logistic Regression-HSMM-based Heart Sound Segmentation," IEEE Trans. Biomed. Eng., In Press, 2015.

5. Feature generation method: 
STFT: MATLAB, code :
[s, F, T, P] = spectrogram(origin_signal,80,60,80,1000);
s = abs(s);
The final s is the STFT matrix;
Envelopes : The function in https://physionet.org/content/hss/1.0/ generates the features;
1D-CNN: Keras code:
model.add(keras.layers.Conv1D(filters=80, kernel_size=80, strides=20,
Padding=’same’, input_shape=(maxlen, 1)))

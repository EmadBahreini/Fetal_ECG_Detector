% Define import options
opts = detectImportOptions('/Users/emadbahreini/University/BiomedicalSignalProcessing/Project/set-a-text/a01.csv');
opts.DataLines = [3 Inf]; % Skip the first two rows
opts.VariableNames = {'time', 'channel1', 'channel2', 'channel3', 'channel4'};

% Read CSV file
abdominal_ecg = readtable('/Users/emadbahreini/University/BiomedicalSignalProcessing/Project/set-a-text/a01.csv', opts);

% Load QRS positions
QRSpositions = dlmread('/Users/emadbahreini/University/BiomedicalSignalProcessing/Project/set-a-text/a01.fqrs.txt');


% Plot the ECG signal
figure;
plot(abdominal_ecg.time, abdominal_ecg.channel1);
title('Abdominal ECG Signal');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

clc, clear, close all

%% load results
list_names = {'011PFNOM.mat', '131EGLPM.mat', '311CLNOM.mat', 'ALVARADO.mat'};

for j = 1:3

    r = load(list_names{j} );

    %% plot waveforms
    figure()
    for i = 1:3
        subplot(3, 1, i)
        plot(r.esrc(i, :), "linewidth", 2)
        hold on
        plot(r.src(i, :), 'linewidth', 2)
    end

    %% plot spectrograms
    figure()
    for i = 1:3
        subplot(3,2,2*i-1)
            spectrogram(r.esrc(i,:),  hanning(1024), 512, 'yaxis')

        subplot(3,2,2*i)
            spectrogram(r.src(i,:), hanning(1024), 512, 'yaxis')
    end

    %% compute evaluation measures
    [SDR, SIR, SAR, perm] = bss_eval_sources(r.esrc, r.src);

    if j == 1
        sdr = SDR;
        sir = SIR;
        sar = SAR;
    else
        sdr = cat(1, sdr, SDR);
        sir = cat(1, sir, SIR);
        sar = cat(1, sar, SAR);
    end

    %% visualize metrics
    fprintf(1, strcat(list_names{j}, '\n'))
    disp(SDR)
    disp(SIR)
    disp(SAR)

    fprintf(1, 'Average SDR = %f [dB]\n', mean(SDR));
    fprintf(1, 'Average SIR = %f [dB]\n', mean(SIR));
    fprintf(1, 'Average SAR = %f [dB]\n', mean(SAR));

    for i = 1:3
        aux = extractBefore(list_names{j}, ".mat");
        aux = strcat(aux, "_source_", num2str(i));
        filename = strcat(aux, ".wav");
        audiowrite(char(filename), r.esrc(i,:), 16000)
    end

end
fprintf(1, 'final results: \n')
fprintf(1, 'Average SDR = %f [dB]\n', round(mean(sdr), 1));
fprintf(1, 'Average SIR = %f [dB]\n', round(mean(sir), 1));
fprintf(1, 'Average SAR = %f [dB]\n', round(mean(sar), 1));

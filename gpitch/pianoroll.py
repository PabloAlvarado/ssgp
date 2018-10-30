import numpy as np
import gpitch
import pandas as pd


class Pianoroll:
    def __init__(self, path, filename=None, fs=20, duration=10.):
        self.path = path
        self.duration = duration
        self.fs = fs
        self.xn = int(round(duration*fs))
        self.x = np.linspace(0., (self.xn -1.)/self.fs, self.xn).reshape(-1, 1)
        self.pr_dic = dict([(str(i), np.zeros((self.xn, 1))) for i in range(21, 109)])

        if filename is None:
            self.name = "unnamed"
        else:
            self.name = gpitch.load_filenames(directory=path, pattern=filename.strip('.wav'), pitches=None,
                                              ext='.txt')[0]

        aux = pd.read_table(self.path + self.name)
        self.pr_pandas = aux[aux["OnsetTime"] < self.duration]

        self.pitch_list = list(set(self.pr_pandas.MidiPitch.tolist()))
        self.pitch_list.sort()

        for i in range(len(self.pitch_list)):
            aux = self.pr_pandas[self.pr_pandas.MidiPitch == self.pitch_list[i]]
            onset = aux.OnsetTime.tolist()
            offset = aux.OffsetTime.tolist()

            key = str(self.pitch_list[i])
            for j in range(len(onset)):
                self.pr_dic[key][(onset[j] <= self.x) & (self.x < offset[j])] = 1.

        self.matrix = []
        for pitch in range(21, 109):
            self.matrix.append(self.pr_dic[str(pitch)].copy())

        self.matrix = np.asarray(self.matrix).reshape(88, -1)
        self.matrix = np.flipud(self.matrix)




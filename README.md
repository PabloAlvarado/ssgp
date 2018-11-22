# SSGP
## Source separation Gaussian process (SSGP) models for audio signals, using variational sparse GPs.
Code used to get results in the paper: *Sparse Gaussian process Audio Source Separation Using Spectrum Priors in the Time-Domain* [pdf][1]. Submitted to 44th International Conference on Acoustics, Speech, and Signal Processing, ICASSP 2019.


This implementation uses [GPflow][2] version 0.5, and [gpitch][3] version 2.0 (a copy is provided in this repository).

### Listen to the reconstructed sources!
The audio files (.wav) of the reconstructed sources using the proposed approach are avaialble [here][4].

[1]: https://arxiv.org/abs/1810.12679
[2]: https://github.com/GPflow/GPflow
[3]: https://github.com/PabloAlvarado/gpitch
[4]: https://sites.google.com/site/paalvaradoduran/ssgp

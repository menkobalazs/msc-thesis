# Detection of alternative bases in nanopore signals using machine learning methods
## Balázs Menkó (ELTE Physics MSc, Scientific Data Analytics and Modeling Specialization)
## Topic leader: István Ervin Csabai (ELTE, Department of Physics of Complex Systems)

### Abstract
In the past decade, machine learning has become an indispensable tool in scientific research, much like how computing technologies revolutionized science in previous years. One of the main advantages of artificial intelligence is its ability to learn from and draw conclusions based on massive datasets - far beyond human capabilities. As a result, it is increasingly used across many fields of natural science, with biology seeing particularly rapid adoption.


One of the revolutionary advancements in DNA sequencing technology is nanopore sequencing. In this method, a DNA strand passes through a tiny pore, modulating the ionic current that flows through it. As different nucleotides affect the current strength in distinct ways, the resulting signal fluctuations can be used to infer the sequence of bases. Machine learning algorithms are employed to accurately interpret this data.


The thymine found in DNA is very similar to the uracil found in RNA, differing by only a single methyl group. However, this difference can play an important role in the nanopore signal, making it possible to detect whether DNA might contain uracil using machine learning algorithms. In my thesis, I present this problem.


### The repository
This repository contains our work mainly in Python notebook format. We worked in the [K8plex](https://k8plex-veo.vo.elte.hu/) system of ELTE.

Folders:
- `01_generate_k-mer_reads`: Generate and analyze $k$-mer reads
- `02_histograms_from_current`: Analysing raw signal without basecall
- `03_hist_from_basecalled_signal`: Analysing raw signal with basecall
- `04_find_uracil_alone`: Find T/U free region in the sequence
- `05_distance_dependency_of_k-mers`: Analyze T/U free region in the sequence
- `06_model`: Setup [DeepMod2](https://github.com/WGLab/DeepMod2), [TandemMod](https://github.com/yulab2021/TandemMod)  and [Uncalled4](https://github.com/skovaka/uncalled4) packages and test them
- `07_new_dataset`: Analyze and align the synthetic dataset with [Minimap2](https://github.com/lh3/minimap2) [Bowtie2](https://github.com/BenLangmead/bowtie2) and heptamer similarity search algorithm


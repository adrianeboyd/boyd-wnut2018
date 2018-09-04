# Low Resource Grammatical Error Correction Using Wikipedia Edits

Please cite the following publication:

```
@InProceedings{boyd2018wnut,
  author    = {Adriane Boyd},
  title     = {Using Wikipedia Edits in Low Resource Grammatical Error Correction},
  booktitle = {Proceedings of the 4th Workshop on Noisy User-generated Text},
  publisher = {Association for Computational Linguistics},
  year      = {2018},
}
```

To replicate the experiments in the paper, install mlconvgec2018 and 
download the data and models as described below.

To generate grammatical error correction data using the method presented 
in the paper, see [Generating GEC Data with Wiki Edits and 
ERRANT](#generating-gec-data-with-wiki-edits-and-errant).

## Install mlconvgec2018

Clone mlconvgec2018:

```
git clone https://github.com/nusnlp/mlconvgec2018/
```

The experiments were performed using commit 
[95725921](https://github.com/nusnlp/mlconvgec2018/tree/95725921fcdc7604a15aad0f9781d0cd9c902400).

Follow installation instructions in `mlconvgec2018/README.md`.

Install moses (the experiments used commit 
[03578921c](https://github.com/moses-smt/mosesdecoder/tree/03578921cc1a03402c601eb9d21f95f8228001fe)) 
and [m2scorer 
2.3](https://github.com/nusnlp/m2scorer/archive/version3.2.tar.gz).

## Download Data

Download the Falko-MERLIN GEC Corpus and German Wikipedia data:
[data.tar.gz](http://www.sfs.uni-tuebingen.de/~adriane/download/wnut2018/data.tar.gz)

Run `./prepare-training-data.sh` in `data/` to generate the training 
files for the experiment.

### Falko / MERLIN Corpora

The original corpora are available at:

- [Falko](https://www.linguistik.hu-berlin.de/de/institut/professuren/korpuslinguistik/forschung/falko/zugang)
- [MERLIN](http://hdl.handle.net/20.500.12124/6)

Tables linking the Falko/MERLIN sentence pairs to their text IDs from 
the original corpora are in `data/source/`. For both corpora, the `ctok` 
(original) and `ZH1`/`TH1` (correction) layers were extracted for the
Falko-MERLIN GEC Corpus.

### Data Licenses

- Falko: CC BY 3.0
- MERLIN: CC BY-SA 4.0
- Wikipedia: CC BY-SA 3.0

## Download Models

Download the BPE and embeddings models: 
[models.tar.gz](http://www.sfs.uni-tuebingen.de/~adriane/download/wnut2018/models.tar.gz)

Due to its size, the language model is not provided for download.

## Run Experiment

Copy `scripts/run_exp.sh` to `mlconvgec2018/training/` and adjust the 
paths and settings at the beginning of the script.

Adjust the paths in `mlconvgec2018/paths.sh`.

The script `run_exp.sh` is used with a single argument to run the 
preprocessing, training, and evaluation steps for one training data set, 
e.g.:

```
./run_exp.sh fm-train-wiki.100K
```

where `fm-train-wiki.100K.src` and `fm-train-wiki.100K.trg` can be found 
in `DATA_DIR` in `paths.sh`.

The corresponding test output is in a timestamped directory, e.g.:

```
fm-train-wiki.100K-2018-08-27-0935-test-dim500-bpe30000-seed1000
```

under:

```
output.tok.txt
output.reranked.eo.tok.txt
output.reranked.eolm-lm.binary.tok.txt
output.reranked.lm-lm.binary.tok.txt
output.reranked.lmnorm-lm.binary.tok.txt
```

and the m2scorer evaluations in the corresponding `.m2scorer` files:

```
output.tok.txt.m2scorer
output.reranked.eo.tok.txt.m2scorer
output.reranked.eolm-lm.binary.tok.txt.m2scorer
output.reranked.lm-lm.binary.tok.txt.m2scorer
output.reranked.lmnorm-lm.binary.tok.txt.m2scorer
```

## Generating GEC Data with Wiki Edits and ERRANT

The following sections describe how German Wikipedia edits were 
extracted from the Wikipedia revision history and filtered using ERRANT.

If necessary, initialize the submodules containing ERRANT and WikiEdits:

```
git submodule update --init --recursive
```

### Wiki Edits

Sentence pairs with small numbers of edits were extracted from Wikipedia 
revision history using [Wiki Edits](https://github.com/snukky/wikiedits) 
with minor modifications to support German available in the submodule 
`wikiedits`.

Extract Wikipedia edits from the Wikipedia history dumps with a maximum 
of 60 words per sentence, e.g.:

```
7zr e -so dewiki-20180601-pages-meta-history1.xml-p1p3342.7z | ./bin/wiki_edits.py -l german -t --max-words 60 > dewiki-20180601-pages-meta-history1.xml-p1p3342.wikiedits.60
```

### ERRANT

#### Install ERRANT

The submodule `errant` contains the extensions to ERRANT for German that 
were used for these experiments.

Install spacy and download the German models:

```
pip install-U spacy==2.0.11
python -m spacy download de
```

Install [TreeTagger](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/) under `errant/resources/tree-tagger-3.2` 
and copy the German model `german-utf8.par` (download: 
[german-par-linux-3.2-utf8.bin.gz](http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/german-par-linux-3.2-utf8.bin.gz)) 
to `errant/resources/tree-tagger-3.2/lib/`.

Install treetaggerwrapper:

```
pip install treetaggerwrapper
```

#### Using ERRANT for German

Profile the tokenized `fm-train` data, specifying German with the new 
option `-lang de`:

```
python parallel_to_m2.py -orig fm-train.src -cor fm-train.trg -out fm-train.m2 -lang de
```

Analyze untokenized Wikipedia edits using the same script with the new 
option `-tok` to have ERRANT also tokenize the input data with the spacy 
tokenizer:

```
python parallel_to_m2.py -orig wiki-unfiltered.src -cor wiki-unfiltered.trg -out wiki-unfiltered.m2 -lang de -tok
```

Filter the Wikipedia edits with reference to the Falko/MERLIN training 
data:

```
python filter_m2.py -filt wiki-unfiltered.m2 -ref fm-train.m2 -out wiki-filtered.m2
```

Convert the filtered wiki m2 back to a plaintext file of parallel 
sentences:

```
python m2_to_parallel.py -m2 wiki-filtered.m2 -out wiki-filtered.src-trg.txt
```

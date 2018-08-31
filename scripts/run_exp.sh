#!/bin/bash

# This script combines and extends training scripts from
# https://github.com/nusnlp/mlconvgec2018/tree/master/training
#
# - preprocess.sh
# - train.sh
# - train_reranker.sh (adding additional reranker conditions)
# - run_trained_model.sh (on test)
# - rerank_output.sh (on test)
# - m2scorer evaluation

# It uses two anaconda (4.4) environments:
# - ANACONDA_FAIRSEQ_ENV: python3 with fairseq
# - ANACONDA_KENLM_ENV: python2 with kenlm for LM reranking
#
# Adjust the environments or comment out the source
# $ANACONDA_BIN_DIR/(de)activate lines as needed for your own environment.

# Significant changes to mlconvgec2018 training scripts are marked inline 
# with "MOD:".

source ../paths.sh

if [ $# -ge 1 ]; then
    train_prefix=$1
else
    echo "Please specify the training data prefix ('train' for train.src / train.trg)"
    echo "Usage: `basename $0` <train_prefix>"
    exit 1
fi

set -x
set -e

#### BEGIN SETTINGS ####

## paths to models and software
ANACONDA_BIN_DIR=/path/to/anaconda3-4.4/bin
ANACONDA_FAIRSEQ_ENV=fairseq
ANACONDA_KENLM_ENV=python2-kenlm
LM=de.deduped.1Blines.binary
LM_PATH=$MODEL_DIR/$LM
SUBWORD_NMT=$SOFTWARE_DIR/subword-nmt
MOSES_PATH=$SOFTWARE_DIR/mosesdecoder
FAIRSEQPY=$SOFTWARE_DIR/fairseq-py
NBEST_RERANKER=$SOFTWARE_DIR/nbest-reranker
M2SCORER_DIR=$SOFTWARE_DIR/m2scorer

bpe_operations=30000
BPE_MODEL=$MODEL_DIR/fm-train-wiki1M.bpe30k.model
EMBED_DIM=500
EMBED_PATH=$MODEL_DIR/wiki.fm-train-wiki1M.dim$EMBED_DIM.bpe.vec

## training parameters
device=0
SEED=1000

## paths to training and development datasets
dev_prefix=fm-dev
test_prefix=fm-test
src_ext=src
trg_ext=trg
m2_ext=m2
train_data_prefix=$DATA_DIR/$train_prefix
dev_data_prefix=$DATA_DIR/$dev_prefix
test_data_prefix=$DATA_DIR/$test_prefix
dev_data_m2=$DATA_DIR/$dev_prefix.m2

## prefixes for all experiment subdirectories
today=`date '+%Y-%m-%d-%H%M'`;
exp_prefix=$1-$today

#### END SETTINGS ####

PROCESSED_DIR=$exp_prefix-processed
OUT_MODEL_DIR=$exp_prefix-models
DATA_BIN_DIR=$PROCESSED_DIR/bin
OUT_DIR=$OUT_MODEL_DIR/mlconv_embed/model-dim$EMBED_DIM-bpe$bpe_operations-seed$SEED

######################
# subword segmentation
mkdir -p $PROCESSED_DIR
# MOD: potentially small training corpora are not sufficient for BPE model,
# so model is provided separately
#cat $train_data_prefix.$trg_ext | $SUBWORD_NMT/learn_bpe.py -s $bpe_operations > $PROCESSED_DIR/models/bpe_model/train.bpe.model
$SCRIPTS_DIR/apply_bpe.py -c $BPE_MODEL < $train_data_prefix.$src_ext > $PROCESSED_DIR/train.all.src
$SCRIPTS_DIR/apply_bpe.py -c $BPE_MODEL < $train_data_prefix.$trg_ext > $PROCESSED_DIR/train.all.trg
$SCRIPTS_DIR/apply_bpe.py -c $BPE_MODEL < $dev_data_prefix.$src_ext > $PROCESSED_DIR/dev.src
$SCRIPTS_DIR/apply_bpe.py -c $BPE_MODEL < $dev_data_prefix.$trg_ext > $PROCESSED_DIR/dev.trg
cp $dev_data_m2 $PROCESSED_DIR/dev.m2
cp $dev_data_prefix.$src_ext $PROCESSED_DIR/dev.input.txt

##########################
#  getting annotated sentence pairs only
# MOD: use all training data, not only instances with corrections
#python $SCRIPTS_DIR/get_diff.py  $PROCESSED_DIR/train.all src trg > $PROCESSED_DIR/train.annotated.src-trg
#cut -f1  $PROCESSED_DIR/train.annotated.src-trg > $PROCESSED_DIR/train.src
#cut -f2  $PROCESSED_DIR/train.annotated.src-trg > $PROCESSED_DIR/train.trg

cp $PROCESSED_DIR/train.all.src $PROCESSED_DIR/train.src
cp $PROCESSED_DIR/train.all.trg $PROCESSED_DIR/train.trg

source $ANACONDA_BIN_DIR/deactivate

source $ANACONDA_BIN_DIR/activate $ANACONDA_FAIRSEQ_ENV

#########################
# preprocessing
python $FAIRSEQPY/preprocess.py --source-lang src --target-lang trg --trainpref $PROCESSED_DIR/train --validpref $PROCESSED_DIR/dev --testpref  $PROCESSED_DIR/dev --nwordssrc $bpe_operations --nwordstgt $bpe_operations --destdir $PROCESSED_DIR/bin

mkdir -p $OUT_DIR

PYTHONPATH=$FAIRSEQPY:$PYTHONPATH CUDA_VISIBLE_DEVICES=$device python $FAIRSEQPY/train.py --save-dir $OUT_DIR --encoder-embed-dim $EMBED_DIM --encoder-embed-path $EMBED_PATH --decoder-embed-dim $EMBED_DIM --decoder-embed-path $EMBED_PATH --decoder-out-embed-dim $EMBED_DIM --dropout 0.2 --clip-norm 0.1 --lr 0.25 --min-lr 1e-4 --encoder-layers '[(1024,3)] * 7' --decoder-layers '[(1024,3)] * 7' --momentum 0.99 --max-epoch 100 --batch-size 28 --no-progress-bar --seed $SEED $DATA_BIN_DIR

output_dir=$exp_prefix-output-dim$EMBED_DIM-bpe$bpe_operations-seed$SEED
model_path=$OUT_DIR/checkpoint_best.pt

beam=12
nbest=$beam
threads=12

## setting model paths
if [[ -d "$model_path" ]]; then
    models=`ls $model_path/*.pt | tr '\n' ' ' | sed "s| \([^$]\)| --path \1|g"`
    echo $models
elif [[ -f "$model_path" ]]; then
    models=$model_path
elif [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path"
fi

###############
# training
###############

TRAIN_DIR=$output_dir/training/
mkdir -p $TRAIN_DIR
echo "[weight]" > $TRAIN_DIR/rerank_config.ini
echo "F0= 0.5" >> $TRAIN_DIR/rerank_config.ini
echo "EditOps0= 0.2 0.2 0.2" >> $TRAIN_DIR/rerank_config.ini

echo "[weight]" > $TRAIN_DIR/rerank_config_eolm.ini
echo "F0= 0.5" >> $TRAIN_DIR/rerank_config_eolm.ini
echo "EditOps0= 0.2 0.2 0.2" >> $TRAIN_DIR/rerank_config_eolm.ini
echo "LM0= 0.5" >> $TRAIN_DIR/rerank_config_eolm.ini
echo "WordPenalty0= -1" >> $TRAIN_DIR/rerank_config_eolm.ini

featstring="EditOps(name='EditOps0')"
featstringeolm="EditOps(name='EditOps0'), LM('LM0', '$LM_PATH', normalize=False), WordPenalty(name='WordPenalty0')"

## MOD: additional reranking options (lm: lm only, lmnorm: normalized lm only)
echo "[weight]" > $TRAIN_DIR/rerank_config_lm.ini
echo "F0= 0.5" >> $TRAIN_DIR/rerank_config_lm.ini
echo "LM0= 0.5" >> $TRAIN_DIR/rerank_config_lm.ini
echo "WordPenalty0= -1" >> $TRAIN_DIR/rerank_config_lm.ini
featstringlm="LM('LM0', '$LM_PATH', normalize=False), WordPenalty(name='WordPenalty0')"
featstringlmnorm="LM('LM0', '$LM_PATH', normalize=True), WordPenalty(name='WordPenalty0')"

source $ANACONDA_BIN_DIR/deactivate

source $ANACONDA_BIN_DIR/activate $ANACONDA_FAIRSEQ_ENV

$SCRIPTS_DIR/apply_bpe.py -c $BPE_MODEL < $PROCESSED_DIR/dev.input.txt > $output_dir/dev.input.bpe.txt

CUDA_VISIBLE_DEVICES=$device python $FAIRSEQPY/generate.py --no-progress-bar --path $models --beam $beam --nbest $beam --interactive --workers $threads $PROCESSED_DIR/bin < $output_dir/dev.input.bpe.txt > $output_dir/dev.output.bpe.nbest.txt

source $ANACONDA_BIN_DIR/deactivate

source $ANACONDA_BIN_DIR/activate $ANACONDA_KENLM_ENV

# reformating the nbest file
$SCRIPTS_DIR/nbest_reformat.py -i $output_dir/dev.output.bpe.nbest.txt --debpe > $output_dir/dev.output.tok.nbest.reformat.txt

reranker_feats=eo

# augmenting the dev nbest
$NBEST_RERANKER/augmenter.py -s $PROCESSED_DIR/dev.input.txt -i $output_dir/dev.output.tok.nbest.reformat.txt -o $output_dir/dev.output.tok.nbest.reformat.augmented.txt -f "$featstring"

# training the nbest to obtain the weights
$NBEST_RERANKER/train.py -i $output_dir/dev.output.tok.nbest.reformat.augmented.txt -r $PROCESSED_DIR/dev.m2 -c $TRAIN_DIR/rerank_config.ini --threads 12 --tuning-metric m2 --predictable-seed -o $TRAIN_DIR --moses-dir $MOSES_PATH --no-add-weight

cp $TRAIN_DIR/weights.txt $output_dir/weights-$reranker_feats.txt

reranker_weights=$output_dir/weights-$reranker_feats.txt

reranker_feats=eolm

# augmenting the dev nbest
$NBEST_RERANKER/augmenter.py -s $PROCESSED_DIR/dev.input.txt -i $output_dir/dev.output.tok.nbest.reformat.txt -o $output_dir/dev.output.tok.nbest.reformat.augmented.txt -f "$featstringeolm"

# training the nbest to obtain the weights
$NBEST_RERANKER/train.py -i $output_dir/dev.output.tok.nbest.reformat.augmented.txt -r $PROCESSED_DIR/dev.m2 -c $TRAIN_DIR/rerank_config_eolm.ini --threads 12 --tuning-metric m2 --predictable-seed -o $TRAIN_DIR --moses-dir $MOSES_PATH --no-add-weight

cp $TRAIN_DIR/weights.txt $output_dir/weights-$reranker_feats-$LM.txt

reranker_weights_eolm=$output_dir/weights-$reranker_feats-$LM.txt

## MOD: additional reranker training

reranker_feats=lm

# augmenting the dev nbest
$NBEST_RERANKER/augmenter.py -s $PROCESSED_DIR/dev.input.txt -i $output_dir/dev.output.tok.nbest.reformat.txt -o $output_dir/dev.output.tok.nbest.reformat.augmented.txt -f "$featstringlm"

# training the nbest to obtain the weights
$NBEST_RERANKER/train.py -i $output_dir/dev.output.tok.nbest.reformat.augmented.txt -r $PROCESSED_DIR/dev.m2 -c $TRAIN_DIR/rerank_config_lm.ini --threads 12 --tuning-metric m2 --predictable-seed -o $TRAIN_DIR --moses-dir $MOSES_PATH --no-add-weight

cp $TRAIN_DIR/weights.txt $output_dir/weights-$reranker_feats-$LM.txt

reranker_weights_lm=$output_dir/weights-$reranker_feats-$LM.txt

reranker_feats=lm

# augmenting the dev nbest
$NBEST_RERANKER/augmenter.py -s $PROCESSED_DIR/dev.input.txt -i $output_dir/dev.output.tok.nbest.reformat.txt -o $output_dir/dev.output.tok.nbest.reformat.augmented.txt -f "$featstringlmnorm"

# training the nbest to obtain the weights
$NBEST_RERANKER/train.py -i $output_dir/dev.output.tok.nbest.reformat.augmented.txt -r $PROCESSED_DIR/dev.m2 -c $TRAIN_DIR/rerank_config_lm.ini --threads 12 --tuning-metric m2 --predictable-seed -o $TRAIN_DIR --moses-dir $MOSES_PATH --no-add-weight

cp $TRAIN_DIR/weights.txt $output_dir/weights-$reranker_feats-norm-$LM.txt

reranker_weights_lmnorm=$output_dir/weights-$reranker_feats-norm-$LM.txt

source $ANACONDA_BIN_DIR/deactivate

input_file=$test_data_prefix.$src_ext
eval_file=$test_data_prefix.$m2_ext
output_dir=$exp_prefix-test-dim$EMBED_DIM-bpe$bpe_operations-seed$SEED
device=0

if [[ -d "$model_path" ]]; then
    models=`ls $model_path/*pt | tr '\n' ' ' | sed "s| \([^$]\)| --path \1|g"`
    echo $models
elif [[ -f "$model_path" ]]; then
    models=$model_path
elif [[ ! -e "$model_path" ]]; then
    echo "Model path not found: $model_path"
fi

beam=12
nbest=$beam
threads=12

mkdir -p $output_dir
$SCRIPTS_DIR/apply_bpe.py -c $BPE_MODEL < $input_file > $output_dir/input.bpe.txt

source $ANACONDA_BIN_DIR/activate $ANACONDA_FAIRSEQ_ENV

# running fairseq on the test data
CUDA_VISIBLE_DEVICES=$device python $FAIRSEQPY/generate.py --no-progress-bar --path $models --beam $beam --nbest $beam --interactive --workers $threads $PROCESSED_DIR/bin < $output_dir/input.bpe.txt > $output_dir/output.bpe.nbest.txt

# getting best hypotheses
cat $output_dir/output.bpe.nbest.txt | grep "^H"  | python -c "import sys; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if(i%$nbest == 0) ]); print(x)" | cut -f3 > $output_dir/output.bpe.txt

# debpe
cat $output_dir/output.bpe.txt | sed 's|@@ ||g' | sed '$ d' > $output_dir/output.tok.txt

source $ANACONDA_BIN_DIR/deactivate

source $ANACONDA_BIN_DIR/activate $ANACONDA_KENLM_ENV

$SCRIPTS_DIR/nbest_reformat.py -i $output_dir/output.bpe.nbest.txt --debpe > $output_dir/output.tok.nbest.reformat.txt
$NBEST_RERANKER/augmenter.py -s $input_file -i $output_dir/output.tok.nbest.reformat.txt -o $output_dir/output.tok.nbest.reformat.augmented.txt -f "$featstring"
$NBEST_RERANKER/rerank.py -i $output_dir/output.tok.nbest.reformat.augmented.txt -w $reranker_weights -o $output_dir --clean-up
mv $output_dir/output.tok.nbest.reformat.augmented.txt.reranked.1best $output_dir/output.reranked.eo.tok.txt

$NBEST_RERANKER/augmenter.py -s $input_file -i $output_dir/output.tok.nbest.reformat.txt -o $output_dir/output.tok.nbest.reformat.augmented.txt -f "$featstringeolm"
$NBEST_RERANKER/rerank.py -i $output_dir/output.tok.nbest.reformat.augmented.txt -w $reranker_weights_eolm -o $output_dir --clean-up
mv $output_dir/output.tok.nbest.reformat.augmented.txt.reranked.1best $output_dir/output.reranked.eolm-$LM.tok.txt

$NBEST_RERANKER/augmenter.py -s $input_file -i $output_dir/output.tok.nbest.reformat.txt -o $output_dir/output.tok.nbest.reformat.augmented.txt -f "$featstringlm"
$NBEST_RERANKER/rerank.py -i $output_dir/output.tok.nbest.reformat.augmented.txt -w $reranker_weights_lm -o $output_dir --clean-up
mv $output_dir/output.tok.nbest.reformat.augmented.txt.reranked.1best $output_dir/output.reranked.lm-$LM.tok.txt

$NBEST_RERANKER/augmenter.py -s $input_file -i $output_dir/output.tok.nbest.reformat.txt -o $output_dir/output.tok.nbest.reformat.augmented.txt -f "$featstringlmnorm"
$NBEST_RERANKER/rerank.py -i $output_dir/output.tok.nbest.reformat.augmented.txt -w $reranker_weights_lmnorm -o $output_dir --clean-up
mv $output_dir/output.tok.nbest.reformat.augmented.txt.reranked.1best $output_dir/output.reranked.lmnorm-$LM.tok.txt

# run potentially slow evaluations in parallel in the background

$M2SCORER_DIR/m2scorer $output_dir/output.tok.txt $eval_file > $output_dir/output.tok.txt.m2scorer &!
$M2SCORER_DIR/m2scorer $output_dir/output.reranked.eo.tok.txt $eval_file > $output_dir/output.reranked.eo.tok.txt.m2scorer &!
$M2SCORER_DIR/m2scorer $output_dir/output.reranked.eolm-$LM.tok.txt $eval_file > $output_dir/output.reranked.eolm-$LM.tok.txt.m2scorer &!
$M2SCORER_DIR/m2scorer $output_dir/output.reranked.lm-$LM.tok.txt $eval_file > $output_dir/output.reranked.lm-$LM.tok.txt.m2scorer &!
$M2SCORER_DIR/m2scorer $output_dir/output.reranked.lmnorm-$LM.tok.txt $eval_file > $output_dir/output.reranked.lmnorm-$LM.tok.txt.m2scorer &!

source $ANACONDA_BIN_DIR/deactivate

exit 0

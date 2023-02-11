#!/usr/bin/env bash

model_dir=${1}
batch_size=${2}
#shift 2
#model_path=${*}
model_step=${3}


model_path="${model_dir}/md_step_${model_step}.pt"
save_path="${model_dir}/test_out_${model_step}.jsonl"
echo "model_path: ${model_path}"
echo "saving to ${save_path}"

echo "=========================================================================";
python translate_simple.py \
-src ../datasets/test_dec.src \
-output "${save_path}" \
-model "${model_path}" \
-replace_unk -gpu 0 \
-batch_size "${batch_size}"

python cal_acc.py \
 ../datasets/test_dec_raw.jsonl \
"${save_path}"

python exp_postproc.py \
"${save_path}" \
"${save_path}-tmp"

perl ./tools/multi-bleu.perl ../datasets/test_dec.gt < "${save_path}-tmp"
rm "${save_path}-tmp"
echo "========================================================================="





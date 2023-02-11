#!/usr/bin/env bash

model_dir=${1}
cls_path=${2}
ppl_path=${3}
pos_path=${4}
neg_path=${5}
num_ckpts=${6:-40}
gap_steps=${7:-5000}
gpu=${8:-0}

for (( i = 0; i < "$num_ckpts"; i ++ ))
do
  cur_step=$[(1 + ${i}) * ${gap_steps}];
  echo "=========================================================================";
  echo "CKPT :: ${cur_step}";
  python translate_wh.py \
  -src "${pos_path}" \
  -tgt "${pos_path}" \
  -output "${model_dir}/test_out_p2n_${cur_step}.txt" \
  -desired_style 0 \
  -model_path "${model_dir}/md_${cur_step}.pt" \
  -vocab_opt_path "${model_dir}/vocab_model_opt.pt" \
  -replace_unk -gpu "${gpu}" -beam_size 30;
  python translate_wh.py \
  -src "${neg_path}" \
  -tgt "${neg_path}" \
  -output "${model_dir}/test_out_n2p_${cur_step}.txt" \
  -desired_style 1 \
  -model_path "${model_dir}/md_${cur_step}.pt" \
  -vocab_opt_path "${model_dir}/vocab_model_opt.pt" \
  -replace_unk -gpu "${gpu}" -beam_size 30;
  echo "========================================================================="
done;


for (( i = 0; i < "$num_ckpts"; i ++ ))
do
  cur_step=$[(1 + ${i}) * ${gap_steps}];
  echo "=========================================================================";
  echo "CKPT :: ${cur_step}";
  python cal_acc.py "${cls_path}" "${model_dir}/test_out_p2n_${cur_step}.txt" 0
  python cal_acc.py "${cls_path}" "${model_dir}/test_out_n2p_${cur_step}.txt" 1
  python cal_ppl.py "${ppl_path}" "${model_dir}/test_out_p2n_${cur_step}.txt"
  python cal_ppl.py "${ppl_path}" "${model_dir}/test_out_n2p_${cur_step}.txt"
  perl tools/multi-bleu.perl "${pos_path}" < "${model_dir}/test_out_p2n_${cur_step}.txt"
  perl tools/multi-bleu.perl "${neg_path}" < "${model_dir}/test_out_n2p_${cur_step}.txt"
  echo "========================================================================="
done;





python -u main.py \
--gpus 2 \
--n_GPUs 1 \
--model EDSR_e2fif \
--load edsr_e2fif \
--resume -2 \
--test_only \
--binary_mode binary \
--dir_data ./sr_data/ \
--epochs 300 \
--decay 200 \
--lr 2e-4 \
--data_test Set5+Set14+Urban100+BSDS100+Manga109 \
--scale 2 \
--n_resblocks 16 \
--n_feats 64 \
--res_scale 1 \
--n_colors 1

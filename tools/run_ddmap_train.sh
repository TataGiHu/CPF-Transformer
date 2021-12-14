cd ..
export CUDA_VISIBLE_DEVICES=0


python tools/train.py \
	--config_file ./config/config_ddmap_normal.py \
	--dist False


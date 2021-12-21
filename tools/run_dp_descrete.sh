cd ..
export CUDA_VISIBLE_DEVICES=0,1,2
python tools/train.py \
	--config_file ./config/config_ddmap_RE.py \
	--dist False

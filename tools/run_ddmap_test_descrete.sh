cd ..
export CUDA_VISIBLE_DEVICES=0


python tools/test.py \
	--config_file ./config/config_ddmap_RE.py \
	--dist False


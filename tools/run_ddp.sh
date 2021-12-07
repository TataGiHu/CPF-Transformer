export CUDA_VISIBLE_DEVICES=0,1,2
cd ..
python -m torch.distributed.launch --nproc_per_node=3 tools/train.py \
    --config_file ./config/myconfig.py \
    --dist True


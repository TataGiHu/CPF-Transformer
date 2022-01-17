export CUDA_VISIBLE_DEVICES=1,2
cd ..
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py \
    --config_file ./config/config_ddmap_three_queries.py \
    --dist True


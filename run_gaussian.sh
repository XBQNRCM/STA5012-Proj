# baseline result for gaussian setting
python run_eval.py \
    --mode gaussian \
    --metric ker \
    --maps performer power_performer \
    --m 8 16 32 64 128 256 512 1024\
    --n-pairs 10000 \
    --n-trials 5000 \
    --seed 42 \
    --device cuda

python run_eval.py \
    --mode gaussian \
    --metric out \
    --maps performer power_performer \
    --m 8 16 32 64 128 256 512 1024\
    --n-pairs 10000 \
    --n-trials 1 \
    --seed 42 \
    --device cuda
# baseline result for gpt2 setting
python run_eval.py \
    --mode gpt2 \
    --metric ker \
    --maps performer power_performer \
    --m 8 16 32 64 128 256 512 1024 \
    --n-docs 1000 \
    --max-length 128 \
    --layers 2,4,6,8,10 \
    --heads 2,4,6,8,10 \
    --token-pos -2 \
    --n-trials 5000 \
    --seed 42 \
    --device cuda

python run_eval.py \
    --mode gpt2 \
    --metric out \
    --maps performer power_performer \
    --m 8 16 32 64 128 256 512 1024 \
    --n-docs 1000 \
    --max-length 128 \
    --layers 2,4,6,8,10 \
    --heads 2,4,6,8,10 \
    --token-pos -2 \
    --n-trials 1 \
    --seed 42 \
    --device cuda
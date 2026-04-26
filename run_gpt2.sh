# baseline result for gpt2 setting
python run_eval.py \
    --mode gpt2 \
    --maps performer \
    --m 16 24 32 40 48 56 64 72 80 \
    --n-docs 1000 \
    --max-length 512 \
    --layers 2,4,6,8,10 \
    --heads 2,4,6,8,10 \
    --token-pos -2 \
    --n-trials 5000 \
    --seed 42 \
    --device cuda
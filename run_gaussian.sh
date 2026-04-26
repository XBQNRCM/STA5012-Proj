# baseline result for gaussian setting
python run_eval.py \
    --mode gaussian \
    --maps performer \
    --m 16 24 32 40 48 56 64 72 80 \
    --n-pairs 10000 \
    --n-trials 100 \
    --seed 42 \
    --device cuda
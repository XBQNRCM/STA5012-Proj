# baseline result for gaussian setting
python run_eval.py \
    --mode gaussian \
    --metric both \
    --maps performer power_performer \
    --m 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160\
    --n-pairs 10000 \
    --n-trials 5000 \
    --seed 42 \
    --device cuda
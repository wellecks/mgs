#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:rtx8000:1
##SBATCH --constraint=nvlink
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dong.qian@mila.quebec

# 1. Load the required modules
module --quiet load anaconda/3

# 2. Load your environment
conda activate torch181

# 3. Copy your dataset on the compute node
rsync -avz ../datasets/wikitext103_raw_gpt2bpe.pkl $SLURM_TMPDIR

# 4. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
for seed in 101 202 303 404 505;
do
	python test_baseline.py --data_path=$SLURM_TMPDIR/wikitext103_raw_gpt2bpe.pkl --seed=$seed --test_model_path=./st/ST_$seed/ --score_mle_model_path=./mle/MLE_seed_$seed/
done


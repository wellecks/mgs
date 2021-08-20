#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:16GB:1
##SBATCH --constraint=nvlink
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH -o ./log/slurm-%j.out

# 1. Load the required modules
# module --quiet load anaconda/3

# 2. Load your environment
# conda activate torch181
module load python/3.7
if [ ! -d ${HOME}/envs/mgs ]; then
	python -m venv ${HOME}/envs/mgs;
	source ${HOME}/envs/mgs/bin/activate;
	python setup.py develop;
	deactivate;
fi
source ${HOME}/envs/mgs/bin/activate

# 3. Copy your dataset on the compute node
rsync -avz ./datasets/wikitext103_raw_gpt2bpe.pkl $SLURM_TMPDIR

MODEL_NAME=${model_name:="gpt2"}
LOSS=${loss:="mle"}
TPORT=${port:=8001}
EXP_NAME=${exp_name:="wikipedia103/"}"_${MODEL_NAME}_${LOSS}"
OUTPUT_DIR_SUFFIX="${MODEL_NAME}_${LOSS}"
SAVE_BASE_DIR=${save_dir:-"./wikipedia103"}

if [ -d ${SAVE_BASE_DIR} ]; then
	mkdir -p ${SAVE_BASE_DIR}
fi 


cmd="python seq_level/gpt2/train.py --dataset-path=$SLURM_TMPDIR/wikitext103_raw_gpt2bpe.pkl"

if [ ${LOSS} = mle ];
then
	cmd+=" --loss mle --valid-every 5000 --print-every 100 "
else
	GGS_METRIC=${ggs_metric:="lm"}
	MLE_MIX=${mle_mix:=0.1}
	MODEL_LOAD_DIR=${model_dir:="./mle/default/"}
	cmd+=" --ggs-metric ${GGS_METRIC} --model-load-dir=${MODEL_LOAD_DIR} "

	if [ ${LOSS} = pg ];
	then
		PG_NORMALIZE_DISTANCE=${pg_dist:=1}
		PG_BASELINE=${baseline:="avg"}
		cmd+="  --loss pg --pg-normalize-distance=${PG_NORMALIZE_DISTANCE} --pg-mle-mix=${MLE_MIX} --pg-baseline ${PG_BASELINE} "

	elif [ ${LOSS} = mrt ];
	then
		MRT_NORMALIZE_DISTANCE=${mrt_dist:=1}
		MRT_BASELINE=${baseline:="avg"}
		cmd+=" --loss mrt --mrt-normalize-distance=${MRT_NORMALIZE_DISTANCE} --mrt-mle-mix=${MLE_MIX} "
		
	elif [ ${LOSS} = ggs ];
	then
		MGS_BETA=${mgs_beta:=1.0}
		cmd+=" --loss ggs --ggs-beta=${MGS_BETA}"

        if [ -n "${include_mle_grad}" ];
        then
            cmd+=" --include-mle-gradient "
        fi

        if [ -n "${efficient}" ]; 
        then
            cmd+=" --efficient "
						if [ -n "${debug}" ];
						then
							cmd+='--score-network-epochs 1 --initial-train-data-size 5'
						fi
        fi
	else
		echo "Input Is Error."
	fi
fi
TMP_RUN_DIR=${SLURM_TMPDIR}/${OUTPUT_DIR_SUFFIX}

cmd+=" --save-base-dir ${TMP_RUN_DIR}"

pkill -f "port ${TPORT}"
sleep 5
echo "Running Command:"
echo "	$cmd"

if [ -z "${debug}" ]; then
	tensorboard --logdir ${TMP_RUN_DIR} --port ${TPORT} --host localhost &
	# For Tensorboard port forwarding based on https://josephpcohen.com/w/jupyter-notebook-and-hpc-systems/.
	ssh -N -R ${TPORT}:localhost:${TPORT} login-4 &
fi

$cmd

if [ -z "${debug}" ]; then
	rsync -avz ${TMP_RUN_DIR} ${SAVE_BASE_DIR}/${OUTPUT_DIR_SUFFIX}
fi

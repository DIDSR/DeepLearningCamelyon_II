#$ -cwd
#$ -l h_rt=048:00:00
#$ -S /bin/sh
#$ -j y
#$ -pe thread 30
#$ -l h_vmem=16G
#$ -l h=bc165

echo "Running job $JOB_ID on $HOSTNAME"

source /home/weizhe.li/environment_setup.sh

time python /home/weizhe.li/RF_training_batch.py

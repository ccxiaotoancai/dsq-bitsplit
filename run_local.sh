#! /bin/bash
#export PATH="/opt/conda/bin:$PATH"
PATH_ENV=$PATH
## switch to script's absolute path
script_path=$(cd `dirname $0`; pwd)
cd $script_path

status=$(nvidia-smi | grep "No running processes found" | awk '{print $2}')

#if [ ! -z $status ]; then

#****************************************Config Info*******************************************
MASTER_PORT=53177

#PYFACE_CMD="/opt/conda/bin/python"
PYFACE_CMD="/usr/bin/python"

HOST_RUN="10.3.70.42" # IP 端口都没啥用不用改
HOST_REF=""  # "10.3.66.14,10.3.66.16,10.3.66.18"
GPU_LIST="0,1,2,3,4,5,6,7"
HOST_LIST="$HOST_RUN $HOST_REF"
USER="mazhenxin"
prefix="/data1/pyface/example/Metric"     #"/model/$script_path/"
mkdir -p models
mkdir -p dss

export  PYTHONPATH=../..

#**********************************************************************************************
usage_info="
============================================================================
usage: ./run.sh -g <gpu_list> -h [host_list]
example: ./run.sh -g 0,1,2,3 -h 127.0.0.1,127.0.0.2
============================================================================"

# 如果命令行自带参数，则使用命令行的参数
while getopts "g:h:" arg
do
        case $arg in
                g)
                        GPU_LIST=$OPTARG
                        ;;
                h)
                        HOST_LIST=$OPTARG
						HOST_LIST=$(echo $HOST_LIST | awk -F "," '{for(i=1;i<=NF;i++){print $i}}')
                        ;;
        esac
done

echo $HOST_LIST
echo $GPU_LIST

GPU_COUNT=`echo $GPU_LIST | awk -F',' '{print NF}'`
HOST_COUNT=`echo $HOST_LIST | awk -F' ' '{print NF}'`
echo "gpu list: $GPU_LIST"
echo "host count: $HOST_COUNT"
echo "gpu per host: $GPU_COUNT"

export CUDA_VISIBLE_DEVICES=${GPU_LIST}


## get master's hostname and port
if [ $DL_NODE_TYPE ];then
    MASTER_ADDR=`hostname`
else
    MASTER_ADDR=$(echo $HOST_LIST | cut -d " " -f1)
fi

#PRE_TRAINED="/data1/pyface/pyface/pretrained/px70b_stride7_baseA_01_ftnowg_toB1_cls_80000_pyv10.pth"
S0_BEGMODEL="$prefix/models/px70b_init_10000.pth"
S0_ENDMODEL="$prefix/models/px70b_init_160000.pth"
S1_BEGMODEL="$prefix/models/px70b_s-cosface0_cls_10000.pth"
S1_ENDMODEL="$prefix/models/px70b_s-cosface0_cls_80000.pth"
# note 要根据conf修改
S2_BEGMODEL="$prefix/models/px70b_s-tow_clc_10000.pth"
S2_ENDMODEL="$prefix/models/px70b_s-tow_clc_180000.pth"

for i in {1..2}
do
  echo $i
  if [ -f $S2_ENDMODEL ]; then
	  echo "Optimization Finished."
	  exit
	elif [ -f $S2_BEGMODEL ]; then
	  echo "stage2 snapshot"
	  LATEST_MODEL=$(ls -t models/px70b_s-tow_clc*.pth | head -n 1)
	  PARAMETER_LIST="main_train.py --stage stage2 --snapshot $LATEST_MODEL "
	  for m in $HOST_REF; do
		scp $(pwd)/$LATEST_MODEL $USER@${m}:$(pwd)/models/
		scp $(pwd)/*.py $USER@${m}:$(pwd)/
	  done
	elif [ -f $S1_ENDMODEL ]; then
	  echo "stage2 weight"
	  PARAMETER_LIST="main_train.py --stage stage2 --weight $S1_ENDMODEL "
	  for mach in $HOST_REF; do
		scp -r $(pwd)/*.py $USER@${mach}:$(pwd)/
		scp $(pwd)/$S1_ENDMODEL $USER@${mach}:$(pwd)/models/
	  done
	elif [ -f $S1_BEGMODEL ]; then
	  echo "stage1 snapshot"
	  LATEST_MODEL=$(ls -t models/px70b_s-cosface0_cls*.pth | head -n 1)
	  PARAMETER_LIST="main_train.py --stage stage1 --snapshot $LATEST_MODEL "
	  for m in $HOST_REF; do
		scp $(pwd)/$LATEST_MODEL $USER@${m}:$(pwd)/models/
		scp $(pwd)/*.py $USER@${m}:$(pwd)/
	  done
	elif [ -f $S0_ENDMODEL ]; then
	  echo "stage1 weight"
	  PARAMETER_LIST="main_train.py --stage stage1 --weight $S0_ENDMODEL "
	  for mach in $HOST_REF; do
		scp -r $(pwd)/*.py $USER@${mach}:$(pwd)/
		scp $(pwd)/$S0_ENDMODEL $USER@${mach}:$(pwd)/models/
	  done
	elif [ -f $S0_BEGMODEL ]; then
	  echo "stage0 snapshot"
	  LATEST_MODEL=$(ls -t models/px70b_init*.pth | head -n 1)
	  PARAMETER_LIST="main_train.py --stage stage0 --snapshot $LATEST_MODEL "
	  for m in $HOST_REF; do
		scp $(pwd)/$LATEST_MODEL $USER@${m}:$(pwd)/models/
		scp $(pwd)/*.py $USER@${m}:$(pwd)/
	  done
	else
	  echo "stage0 init"
	  PARAMETER_LIST="main_train.py --stage stage0"
	  for mach in $HOST_REF; do
		scp -r $(pwd)/*.py $USER@${mach}:$(pwd)/
	  done
	fi
	
	echo $PARAMETER_LIST
	
	DTM=$(date +%Y%m%dT%H%M%S)
	LOG_PATH=proc_$DTM.log

	if [ $HOST_COUNT -le 1 ]; then
		echo "single-node multi-gpu distributed training"
		$PYFACE_CMD -m torch.distributed.launch --master_port=${MASTER_PORT} --nproc_per_node=${GPU_COUNT} $PARAMETER_LIST --nproc_per_node=${GPU_COUNT} 2>&1 | tee $LOG_PATH
	else
		echo "==> multi-Node multi-process distributed training"
		echo "host list: [$HOST_COUNT] $HOST_LIST"
		echo "master addr： $MASTER_ADDR:$MASTER_PORT"
			
		# start slave
		node_idx=1
		slave_list=$(echo $HOST_LIST | awk -F " " '{for(i=1;i<=NF;i++){print $i}}' | sed  "s/${MASTER_ADDR}//g")
		for slave_addr in ${slave_list}
		do
			echo "==> start slave node $slave_addr"
			ssh $slave_addr bash -c " 'export PATH=${PATH_ENV}; cd $script_path; $PYFACE_CMD -m torch.distributed.launch --nproc_per_node=${GPU_COUNT} --nnodes=${HOST_COUNT} \
				--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --node_rank=$node_idx \
				$PARAMETER_LIST &' " &
			let "node_idx=node_idx+1"
		done
		
		# start master
		echo "start master node $MASTER_ADDR"
		node_idx=0
		$PYFACE_CMD -m torch.distributed.launch --nproc_per_node=${GPU_COUNT} --nnodes=${HOST_COUNT} \
			--master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} --node_rank=$node_idx \
			$PARAMETER_LIST 2>&1 | tee $LOG_PATH
	fi
done
exit 1

#These variables need to be defined for running distributed pytorch
numnodes=$1
rank=$2
masterip=$3

export PYTHONPATH=$PYTHONPATH:/home/saghotra/git/SRFlow/

#Kill Currently Running Jobs
sudo -H pkill python3.6

NCCL_SOCKET_IFNAME=^lo,docker0,veth NCCL_DEBUG=INFO \
NCCL_TREE_THRESHOLD=0 \
python3.6 -m torch.distributed.launch \
--nproc_per_node=8 \
--nnodes=$numnodes \
--node_rank=$rank \
--master_addr=$masterip \
--master_port=12345 /home/saghotra/git/SRFlow/code/train.py \
-opt /home/saghotra/git/SRFlow/code/confs/tmp_RRDB_DF2K_4X.yml \
--launcher "pytorch"

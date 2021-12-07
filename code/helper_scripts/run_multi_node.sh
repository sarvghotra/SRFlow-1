WORK_DIR=/home/saghotra/git/SRFlow/code/helper_scripts/

#Script for doing all the setup for the app such as installing dependencies
# SETUPSCRIPT=${WORK_DIR}/distributed_run/node_environment_setup.sh

#Runs on each node to setup distributed Python execution
JOBSCRIPT=${WORK_DIR}/run_on_node.sh

#Training script is executed on each node using job script
APPSCRIPT=$1
JOBDIR=$2   #model dir path

SETUPLOG=/tmp/DistributedSetupLog
RANKSETUPLOG=/tmp/RankSetupLog
RANKERRORLOG=/tmp/RankErrorLog
JOBLOG=/tmp/DistributedJobLogs
JOBERRORLOG=/tmp/DistributedErrorLogs

rm -rf /tmp/Distributed*

#Install Parallel SSH
#echo "Setting up Parallel SSH"
#sudo -H apt-get install pssh

#Run SetupScript on all the nodes
#echo "Running Full Distributed Setup on All Nodes. Check $SETUPLOG"
#parallel-ssh -t 0 -o $SETUPLOG -h /job/mpi-hosts bash $SETUPSCRIPT

#Find appropriate rank for each node
echo "Setting up rank id. Check $RANKSETUPLOG"
#hostname=`hostname -I`
#hostip=`echo $hostname | awk '{print $1}'`

echo "Setting up mpi-hosts. Check $JOBDIR/mpi-hosts"

cat ~/.ssh/config | grep worker- | cut -d' ' -f2 > $JOBDIR/mpi-hosts  #/home/saghotra/models/semantic/interim_albert/capt_qna_1n/mpi-hosts

master_ip=`head -1 $JOBDIR/mpi-hosts`
count=1
sudo -H rm /tmp/host-ranks-master
for host in `cat $JOBDIR/mpi-hosts`
do
    #ip=`grep $host /etc/hosts | awk '{print $1}'`
    if [ "$host" == "$master_ip" ]; then
        rank=0
    else
        rank=$count
        count=$((count+1))
    fi
    ip=`grep -A1 "\<${host}\>" /job/.ssh/config | grep HostName | awk '{print $2}'`
    echo "$ip $host $rank" >> /tmp/host-ranks-master
    echo "$ip $host $rank"
done

echo "/tmp/host-ranks-master:"
cat /tmp/host-ranks-master

parallel-scp -o $RANKSETUPLOG -e $RANKERRORLOG -h $JOBDIR/mpi-hosts /tmp/host-ranks-master /tmp/ip-ranks

#Run the actual job script
echo "Running Job Script. Check $JOBLOG and $JOBERRORLOG"
pssh -x "-tt" -t 0 -o $JOBLOG -e $JOBERRORLOG -h $JOBDIR/mpi-hosts bash $JOBSCRIPT $APPSCRIPT


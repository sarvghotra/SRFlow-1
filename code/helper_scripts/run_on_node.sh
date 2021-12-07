###########################################################
#figure out the rank to use, IP of the master node, and number of nodes
############################################################
hostranks=/tmp/ip-ranks
masterip=""

hostid=`hostname -I | awk '{print $1}'`
rank=0

while read line
do
    ip=`echo $line | awk '{print $1}'`
    currank=`echo $line | awk '{print $3}'`
    #echo "IP  $ip Cur Rank $currank Host ID $hostid"
    if [ $ip == $hostid ]; then
        rank=$currank
    fi
    if [ "$currank" == "0" ]; then
        masterip=$ip
    fi
done < $hostranks
numnodes=`wc -l $hostranks | awk '{print $1}'`
echo "Num Nodes $numnodes Master IP $masterip Rank $rank"
######################################################

#Running the Distributed Python Application Script
APPSCRIPT=$1

bash $APPSCRIPT $numnodes $rank $masterip

echo "Done"

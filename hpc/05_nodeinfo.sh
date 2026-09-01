#!/bin/bash
# Run on the LOGIN node. How many cores/GPUs each node actually has, and what the queues
# will let us ask for -- 30_train.pbs requests ncpus=16 and it must be grantable.
echo "=== nodes ==="
pbsnodes -a 2>/dev/null | grep -E "^[a-zA-Z]|resources_available.(ncpus|ngpus|mem)|resources_assigned.(ncpus|ngpus)|state =|queue =" | head -60
echo
echo "=== queue limits ==="
qstat -Qf 2>/dev/null | grep -E "^Queue|resources_max|resources_default|max_run|max_queued|acl_" 
echo
echo "=== who else is on the gpu queue ==="
qstat -a 2>/dev/null | tail -20

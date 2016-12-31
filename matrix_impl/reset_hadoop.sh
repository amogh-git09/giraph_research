stop-dfs.sh
stop-mapred.sh
start-dfs.sh
start-mapred.sh
hadoop namenode -format
hadoop dfs -copyFromLocal mnist_sub.csv /user/hduser/input/mnist.txt
hadoop dfs -ls /user/hduser/input/

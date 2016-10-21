./gradlew build -x test
rm -f .aggreg*
rm -f aggre*
giraph build/libs/*.jar neural_net.BackwardPropagation -vip src/main/resources/5 -vif neural_net.NeuralNetworkVertexInputFormat -mc neural_net.NumberOfClasses -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error
#giraph build/libs/*.jar neural_net.BackwardPropagation -vip src/main/resources/5 -vif neural_net.NeuralNetworkVertexInputFormat -mc neural_net.NumberOfClasses -c org.apache.giraph.combiner.DoubleSumMessageCombiner -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error
#giraph build/libs/*.jar GiraphHelloWorld -vip src/main/resources/1 -vif SimpleTextVertexInputFormat -mc TotalNumberOfEdgesMC -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error
rm -rf output

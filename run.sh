./gradlew build -x test
rm -f .aggreg*
rm -f aggre*
giraph build/libs/*.jar neural_net.NeuralNetworkHelloWorld -vip src/main/resources/5 -vif neural_net.NeuralNetworkVertexInputFormat -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error
rm -rf output

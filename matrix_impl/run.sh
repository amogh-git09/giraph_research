./gradlew fatjar

#giraph build/libs/*.jar DMWTest -vip src/main/resources/1 -vif org.apache.giraph.io.formats.IntIntNullTextInputFormat -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error

giraph build/libs/*.jar Backpropagation -mc NNMasterCompute -vip src/main/resources/1 -vif NeuralNetworkVectorVertexInputFormat -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error,giraph.workerContextClass=NNWorkerContext

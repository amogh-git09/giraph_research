./gradlew fatjar

#giraph build/libs/*.jar DVWTest -vip src/main/resources/1 -vif org.apache.giraph.io.formats.IntIntNullTextInputFormat -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error

giraph build/libs/*.jar Backpropagation -vip src/main/resources/1 -vif NeuralNetworkVectorVertexInputFormat -w 1 -ca giraph.SplitMasterWorker=false,giraph.logLevel=error

./gradlew fatjar

giraph build/libs/*.jar distributed_net.DistNeuralNet -vip src/main/resources/1 -vif nn_input_format.NeuralNetworkVertexInputFormat -mc master_compute.NNMasterCompute  -w 1 -ca giraph.workerContextClass=worker_context.RedisWorkerContext,giraph.SplitMasterWorker=false,giraph.logLevel=error


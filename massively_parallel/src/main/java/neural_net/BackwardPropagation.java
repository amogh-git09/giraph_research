package neural_net;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.giraph.GiraphRunner;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVector;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class BackwardPropagation extends
        BasicComputation<Text, NeuronValue, DoubleWritable, Text> {

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new GiraphRunner(), args));
    }

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        Logger.i1("SS: " + getSuperstep() + "  Vertex ID: " + vertex.getId());

        String[] tokens = vertex.getId().toString().split(Config.DELIMITER);
        int networkNum = Integer.parseInt(tokens[0]);
        int layerNum = Integer.parseInt(tokens[1]);
        int neuronNum = Integer.parseInt(tokens[2]);

        if (getSuperstep() == 0) {
            if (layerNum == Config.OUTPUT_LAYER) {
                vertex.voteToHalt();
                return;
            }
        }

        if (getSuperstep() >= Config.MAX_ITER - Config.MAX_HIDDEN_LAYER_NUM) {
            //aggregate the weights
            if (networkNum == 1 && layerNum != Config.OUTPUT_LAYER) {
                Logger.d("Aggregating weights for vertex " + vertex.getId());
                String aggName = NumberOfClasses.GetWeightAggregatorName(layerNum, neuronNum);
                DoubleDenseVector weights = getAggregatedValue(aggName);

                //flush the aggregator
                int vecSize = layerNum == Config.INPUT_LAYER ? Config.HIDDEN_LAYER_NEURON_COUNT :
                        Config.OUTPUT_LAYER_NEURON_COUNT;
                DoubleDenseVector vec = new DoubleDenseVector(vecSize);
                for (int v = 0; v < vecSize; v++) {
                    vec.set(v, -weights.get(v));
                }
                aggregate(aggName, vec);

                //set latest weights
                for (Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
                    if (isAnEdgeToNextLayer(e, layerNum)) {
                        Text dstId = e.getTargetVertexId();
                        String[] edgeTokens = dstId.toString().split(Config.DELIMITER);
                        int dstNeuronNum = Integer.parseInt(edgeTokens[2]);

                        Logger.d(String.format("Edge from %s --> %s, weight = %f",
                                vertex.getId(), dstId.toString(), e.getValue().get()));
                        vec.set(dstNeuronNum - 1, e.getValue().get());
                    }
                }

                aggregate(aggName, vec);
            }
        }

        IntWritable state = getAggregatedValue(NumberOfClasses.STATE_ID);

        switch (state.get()) {
            case NumberOfClasses.HIDDEN_LAYER_GENERATION_STATE:
                if (layerNum == Config.MAX_HIDDEN_LAYER_NUM) {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, Config.OUTPUT_LAYER,
                            Config.OUTPUT_LAYER_NEURON_COUNT, neuronNum);
                } else if (layerNum == Config.OUTPUT_LAYER) {
                    if (neuronNum == 1) {
                        turnInputLayerToActive(networkNum);
                        //aggregate number of networks
                        aggregate(NumberOfClasses.NUMBER_OF_NETWORKS_ID, new IntWritable(networkNum));

                        Logger.d("Changing to back edges generation stage");
                        aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACK_EDGES_GENERATION_STATE));
                    }
                } else {
                    //generate next layer's bias unit
                    if (neuronNum == Config.BIAS_UNIT && layerNum != Config.MAX_HIDDEN_LAYER_NUM) {
                        Text id = new Text(networkNum + Config.DELIMITER + (layerNum + 1) + Config.DELIMITER + 0);
                        Logger.d("Generating bias unit " + id);
                        sendMessage(id, new Text(""));
                    }

                    generateEdgesToNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            Config.HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }

                vertex.voteToHalt();
                break;

            case NumberOfClasses.BACK_EDGES_GENERATION_STATE:
                if (layerNum == Config.MAX_HIDDEN_LAYER_NUM) {
                    generateEdgesFromNextLayer(vertex, networkNum, layerNum, Config.OUTPUT_LAYER,
                            Config.OUTPUT_LAYER_NEURON_COUNT, neuronNum);
                } else if (layerNum == Config.OUTPUT_LAYER) {
                    if (neuronNum == 1) {
                        turnInputLayerToActive(networkNum);

                        Logger.d("Changing to forward propagation stage");
                        aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                    }

                    vertex.voteToHalt();
                    break;
                } else {
                    generateEdgesFromNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            Config.HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }


                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACK_EDGES_GENERATION_STATE));
                vertex.voteToHalt();
                break;

            case NumberOfClasses.FORWARD_PROPAGATION_STATE:
                // update the weights
                updateFrontWeights(vertex, layerNum, neuronNum);
                updateBackWeights(vertex, layerNum, neuronNum);

                // set new activation
                if (layerNum != Config.INPUT_LAYER && neuronNum != Config.BIAS_UNIT) {
                    double weightedInput = 0d;

                    for (Text m : messages) {
                        Double input = Double.parseDouble(m.toString());
                        weightedInput += input;
                        Logger.d("message : " + m);
                    }

                    vertex.getValue().setWeightedInput(weightedInput);
                    Double activation = activationFunction(weightedInput);
                    vertex.getValue().setActivation(activation);

                    Logger.d("Weighted Input: " + vertex.getValue().getWeightedInput());
                    Logger.d("Activation: " + vertex.getValue().getActivation());
                }

                // aggregate cost
                if (layerNum == Config.OUTPUT_LAYER) {
                    //flush aggregator (only once)
                    if (networkNum == 1 && neuronNum == 1) {
                        DoubleWritable oldCost = getAggregatedValue(NumberOfClasses.COST_AGGREGATOR);
                        aggregate(NumberOfClasses.COST_AGGREGATOR, new DoubleWritable(-oldCost.get()));
                    }

                    double cost = neuronCost(vertex);
                    Logger.d("Cost frag = " + cost);
                    aggregate(NumberOfClasses.COST_AGGREGATOR, new DoubleWritable(cost));
                }

                // forward propagation
                if (layerNum != Config.OUTPUT_LAYER) {
                    //activate bias unit of next layer
                    if (neuronNum == 1 && layerNum != Config.MAX_HIDDEN_LAYER_NUM)
                        sendMessage(new Text(networkNum + Config.DELIMITER + (layerNum + 1) +
                                Config.DELIMITER + Config.BIAS_UNIT), new Text(""));

                    //set correct activation in case of bias unit
                    if (neuronNum == Config.BIAS_UNIT)
                        vertex.getValue().setActivation(1d);

                    forwardProp(vertex, layerNum);
                } else {
                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACKWARD_PROPAGATION_STATE));
                    break;
                }

                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                vertex.voteToHalt();
                break;

            case NumberOfClasses.BACKWARD_PROPAGATION_STATE:
                if (layerNum != Config.OUTPUT_LAYER) {
                    //flush error aggregator DELTA
                    if (layerNum != Config.OUTPUT_LAYER && networkNum == 1) {
                        Logger.d(String.format("flushing error agg for layerNum: %s, neuronNum: %d\n", layerNum, neuronNum));

                        int nextLayerNeuronCount = Config.HIDDEN_LAYER_NEURON_COUNT;
                        if (layerNum == Config.MAX_HIDDEN_LAYER_NUM)
                            nextLayerNeuronCount = Config.OUTPUT_LAYER_NEURON_COUNT;

                        // reset error accumulator to zero
                        flushErrorAggregator(layerNum, neuronNum, nextLayerNeuronCount);
                    }

                    double weightedError = 0d;
                    DoubleDenseVector errVector;

                    if (layerNum == Config.MAX_HIDDEN_LAYER_NUM)
                        errVector = new DoubleDenseVector(Config.OUTPUT_LAYER_NEURON_COUNT);
                    else
                        errVector = new DoubleDenseVector(Config.HIDDEN_LAYER_NEURON_COUNT);

                    for (Text m : messages) {
                        String[] msgTokens = m.toString().split(Config.DELIMITER);
                        int senderNeuronNum = Integer.parseInt(msgTokens[0]);
                        Double activation = vertex.getValue().getActivation();
                        Double input = Double.parseDouble(msgTokens[1]);
                        Double srcError = Double.parseDouble(msgTokens[2]);

                        errVector.set(senderNeuronNum - 1, activation * srcError);      // to calculate gradient later
                        weightedError += input;
                        Logger.d("message   : " + m);
                    }

                    String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
                    aggregate(aggName, errVector);

                    int size;
                    if (layerNum == Config.MAX_HIDDEN_LAYER_NUM)
                        size = Config.OUTPUT_LAYER_NEURON_COUNT;
                    else
                        size = Config.HIDDEN_LAYER_NEURON_COUNT;
                    Logger.d("errVector : ");
                    for (int i = 0; i < size; i++) {
                        Logger.d(errVector.get(i) + "  ");
                    }

                    if (layerNum != Config.INPUT_LAYER) {
                        Double weightedInput = vertex.getValue().getWeightedInput();
                        Double error = weightedError * activationFunctionDerivative(weightedInput);
                        vertex.getValue().setError(error);
                        Logger.d("Error     : " + error);
                    }
                } else {
                    // output layer error
                    Double error = vertex.getValue().getActivation() - vertex.getValue().getClassFlag();
                    vertex.getValue().setError(error);
                    Logger.d("New error: " + vertex.getValue().getError());
                }

                // backpropagation of error
                if (layerNum != Config.INPUT_LAYER) {
                    if (neuronNum != Config.BIAS_UNIT) {
                        backPropagateError(vertex, layerNum, neuronNum);
                        aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACKWARD_PROPAGATION_STATE));
                    }

                    vertex.voteToHalt();
                } else {
                    //keep input layer active, only change the state
                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                }

                break;
        }
    }

    private void flushErrorAggregator(int layerNum, int neuronNum, int nextLayerNeuronCount) {

        String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
        DoubleDenseVector gradients = getAggregatedValue(aggName);
        DoubleDenseVector canceller = new DoubleDenseVector(nextLayerNeuronCount);

        for (int i = 0; i < nextLayerNeuronCount; i++) {
            Double grad = gradients.get(i);
            canceller.set(i, -grad);
        }

        aggregate(aggName, canceller);
    }

    private double neuronCost(Vertex<Text, NeuronValue, DoubleWritable> vertex) {
        NeuronValue val = vertex.getValue();
        int y = val.getClassFlag();
        double activation = val.getActivation();
        if (activation == 0) {
            System.out.println("\n\n\nACTIVATION ZERO\n\n\n");
            return 0;
        }
        double fragment = y * Math.log(activation) + (1 - y) * Math.log(1 - activation);

        if (fragment > 0) {
            System.out.printf("positive cost found for vertex: %s \nactivation = %f, cost = %f, y = %d\n",
                    vertex.getId(), activation, fragment, y);
        }

        Logger.d(String.format("y: %d, activation: %f\n", y, activation));
        Logger.d(String.format("fragment = %f\n", fragment));
        return fragment;
    }

    private void updateFrontWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        if (layerNum == Config.OUTPUT_LAYER)
            return;

        IntWritable m = getAggregatedValue(NumberOfClasses.NUMBER_OF_NETWORKS_ID);
        String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
        DoubleDenseVector gradients = getAggregatedValue(aggName);

        for (Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
            if (isAnEdgeToNextLayer(e, layerNum)) {
                Text dstId = e.getTargetVertexId();
                String[] edgeTokens = dstId.toString().split(Config.DELIMITER);
                int dstNeuronNum = Integer.parseInt(edgeTokens[2]);

                Logger.d("gradients: ");
                int nextLayerNeuronCnt = layerNum == Config.MAX_HIDDEN_LAYER_NUM ? Config.OUTPUT_LAYER_NEURON_COUNT : Config.HIDDEN_LAYER_NEURON_COUNT;
                for (int i = 0; i < nextLayerNeuronCnt; i++) {
                    Logger.d(gradients.get(i) + "  ");
                }

                Double gradient = gradients.get(dstNeuronNum - 1) / m.get();
                Double old = e.getValue().get();
                Double update = gradient * Config.LEARNING_RATE;

                // gradient descent
                e.getValue().set(old - update);

                Logger.d("Updating front edge " + vertex.getId() + " --> " + e.getTargetVertexId());
                Logger.d(String.format("Old val: %f, gradient: %f, update: %f, new val: %f\n",
                        old, gradient, update, e.getValue().get()));
            }
        }
    }

    private void updateBackWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        if (layerNum == Config.INPUT_LAYER)
            return;

        IntWritable m = getAggregatedValue(NumberOfClasses.NUMBER_OF_NETWORKS_ID);
        int prevLayerNum = layerNum == Config.OUTPUT_LAYER ? Config.MAX_HIDDEN_LAYER_NUM : (layerNum - 1);
        int cnt = 0;

        for (Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
            if (isAnEdgeToPrevLayer(e, layerNum)) {
                Text dstId = e.getTargetVertexId();
                String[] tokens = dstId.toString().split(Config.DELIMITER);
                int dstNeuronNum = Integer.parseInt(tokens[2]);

                String aggName = NumberOfClasses.GetErrorAggregatorName(prevLayerNum, dstNeuronNum);
                DoubleDenseVector gradients = getAggregatedValue(aggName);

                Double gradient = gradients.get(neuronNum - 1) / m.get();
                Double old = e.getValue().get();
                Double update = gradient * Config.LEARNING_RATE;

                Logger.d("Updating back edge " + vertex.getId() + " --> " + e.getTargetVertexId());
                Logger.d(String.format("Old val: %f, gradient: %f, update: %f, new val: %f\n",
                        old, gradient, update, (old - update)));

                // gradient descent
                e.getValue().set(old - update);
            }
        }
    }

    private void backPropagateError(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        for (Edge<Text, DoubleWritable> e : vertex.getEdges()) {

            Text dstId = e.getTargetVertexId();

            if (isAnEdgeToPrevLayer(e, layerNum)) {
                Double weight = e.getValue().get();
                Double error = vertex.getValue().getError();
                Double fragment = weight * error;
                String msg = neuronNum + Config.DELIMITER + fragment + Config.DELIMITER + error;
                sendMessage(dstId, new Text(msg));
                Logger.d(String.format("weight: %f, error: %f\n", weight, error));
                Logger.d("Sending msg to " + dstId + ", msg = " + msg);
            }
        }
    }

    private void forwardProp(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum) {
        for (Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            Text dstId = e.getTargetVertexId();

            if (isAnEdgeToNextLayer(e, layerNum)) {
                Double weight = e.getValue().get();
                Double activation = vertex.getValue().getActivation();
                Double fragment = weight * activation;
                sendMessage(dstId, new Text(fragment + ""));
                Logger.d(String.format("weight: %f, activation: %f\n", weight, activation));
                Logger.d("Sending msg to " + dstId + ", msg = " + fragment);
            }
        }
    }

    private boolean isAnEdgeToNextLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(Config.DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[1]);

        if (dstLayerNum == Config.OUTPUT_LAYER) {
            return layerNum == Config.MAX_HIDDEN_LAYER_NUM;
        } else {
            return (layerNum + 1 == dstLayerNum);
        }
    }

    private boolean isAnEdgeToPrevLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(Config.DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[1]);

        if (dstLayerNum == Config.MAX_HIDDEN_LAYER_NUM) {
            return layerNum == Config.OUTPUT_LAYER;
        } else {
            return (layerNum - 1 == dstLayerNum);
        }
    }

    private void generateEdgesToNextLayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                          int networkNum, int thisLayer, int nextLayer,
                                          int nextLayerCount, int neuronNum) throws IOException {

        // no edges to bias unit
        for (int i = 1; i <= nextLayerCount; i++) {
            DoubleDenseVector weights = getAggregatedValue(
                    NumberOfClasses.GetWeightAggregatorName(thisLayer, neuronNum));
            Double weight = weights.get(i - 1);

            Text dstId = new Text(networkNum + ":" + nextLayer + ":" + i);
            sendMessage(dstId, new Text(""));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(weight));
            addEdgeRequest(vertex.getId(), e);
            Logger.d("Generated edge from " + vertex.getId() + " to " +
                    e.getTargetVertexId() + " with weight " + e.getValue());
        }
    }

    private void generateEdgesFromNextLayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                            int networkNum, int thisLayer, int nextLayer,
                                            int nextLayerCount, int neuronNum) throws IOException {

        for (int i = 1; i <= nextLayerCount; i++) {
            DoubleDenseVector weights = getAggregatedValue(
                    NumberOfClasses.GetWeightAggregatorName(thisLayer, neuronNum));
            Double weight = weights.get(i - 1);

            Text srcId = new Text(networkNum + ":" + nextLayer + ":" + i);
            Edge<Text, DoubleWritable> e = EdgeFactory.create(vertex.getId(), new DoubleWritable(weight));
            addEdgeRequest(srcId, e);
            Logger.d("Generated edge from " + srcId + " to " +
                    e.getTargetVertexId() + " with weight " + e.getValue());

            sendMessage(srcId, new Text(""));       // to activate previous layer
        }

        //activate bias unit of next layer
        Text biasDst = new Text(String.format("%d%s%d%s%d", networkNum, Config.DELIMITER, nextLayer, Config.DELIMITER, Config.BIAS_UNIT));
        sendMessage(biasDst, new Text(""));
    }

    // input x should the weighted input
    private double activationFunction(Double x) {
        Sigmoid sig = new Sigmoid();
        return sig.value(x);
    }

    // input x must be the weighted input
    private double activationFunctionDerivative(Double x) {
        Sigmoid sig = new Sigmoid();
        return sig.value(x) * (1 - sig.value(x));
    }

    private void turnInputLayerToActive(int networkNum) {
        for (int i = 0; i <= Config.INPUT_LAYER_NEURON_COUNT; i++) {
            Text dstId = new Text(String.format("%d:%d:%d", networkNum, Config.INPUT_LAYER, i));
            sendMessage(dstId, new Text(""));
        }
    }
}

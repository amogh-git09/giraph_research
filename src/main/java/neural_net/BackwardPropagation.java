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

    static final int MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
    static final int INPUT_LAYER_NEURON_COUNT = 4;
    static final int HIDDEN_LAYER_NEURON_COUNT = 4;
    static final int OUTPUT_LAYER_NEURON_COUNT = 3;

//    static final int MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
//    static final int INPUT_LAYER_NEURON_COUNT = 2;
//    static final int HIDDEN_LAYER_NEURON_COUNT = 2;
//    static final int OUTPUT_LAYER_NEURON_COUNT = 1;

    static final int BIAS_UNIT = 0;
    static final int INPUT_LAYER = 1;
    static final int OUTPUT_LAYER = -1;
    static final String DELIMITER = ":";
    static final double LEARNING_RATE = 0.1;
    static final int MAX_ITER = 15000;

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new GiraphRunner(), args));
    }

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

//        System.out.println("SS: " + getSuperstep() + "  Vertex ID: " + vertex.getId());

        String[] tokens = vertex.getId().toString().split(DELIMITER);
        int networkNum = Integer.parseInt(tokens[0]);
        int layerNum = Integer.parseInt(tokens[1]);
        int neuronNum = Integer.parseInt(tokens[2]);

        if (getSuperstep() == 0) {
            if (layerNum == NeuralNetworkVertexInputFormat.OUTPUT_LAYER) {
                vertex.voteToHalt();
                return;
            }
        }


        if (getSuperstep() >= MAX_ITER - MAX_HIDDEN_LAYER_NUM) {
            //aggregate the weights
            if (networkNum == 1 && layerNum != OUTPUT_LAYER) {
                System.out.println("Aggregating weights for vertex " + vertex.getId());
                String aggName = NumberOfClasses.GetWeightAggregatorName(layerNum, neuronNum);
                DoubleDenseVector weights = getAggregatedValue(aggName);

                //flush the aggregator
                int vecSize = layerNum == INPUT_LAYER ? HIDDEN_LAYER_NEURON_COUNT : OUTPUT_LAYER_NEURON_COUNT;
                DoubleDenseVector vec = new DoubleDenseVector(vecSize);
                for (int v = 0; v < vecSize; v++) {
                    vec.set(v, -weights.get(v));
                }
                aggregate(aggName, vec);

                //set latest weights
                for (Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
                    if (isAnEdgeToNextLayer(e, layerNum)) {
                        Text dstId = e.getTargetVertexId();
                        String[] edgeTokens = dstId.toString().split(DELIMITER);
                        int dstNeuronNum = Integer.parseInt(edgeTokens[2]);

                        System.out.printf("Edge from %s --> %s, weight = %f\n", vertex.getId(), dstId.toString(), e.getValue().get());
                        vec.set(dstNeuronNum - 1, e.getValue().get());
                    }
                }

                aggregate(aggName, vec);
            }
        }

        IntWritable state = getAggregatedValue(NumberOfClasses.STATE_ID);

        switch (state.get()) {
            case NumberOfClasses.HIDDEN_LAYER_GENERATION_STATE:
                if (layerNum == MAX_HIDDEN_LAYER_NUM) {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, OUTPUT_LAYER,
                            OUTPUT_LAYER_NEURON_COUNT, neuronNum);
                } else if (layerNum == OUTPUT_LAYER) {
                    if (neuronNum == 1) {
                        turnInputLayerToActive(networkNum);
                        //aggregate number of networks
                        aggregate(NumberOfClasses.NUMBER_OF_NETWORKS_ID, new IntWritable(networkNum));

//                        System.out.println("Changing to back edges generation stage");
                        aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACK_EDGES_GENERATION_STATE));
                    }
                } else {
                    //generate next layer's bias unit
                    if (neuronNum == BIAS_UNIT && layerNum != MAX_HIDDEN_LAYER_NUM) {
                        Text id = new Text(networkNum + DELIMITER + (layerNum + 1) + DELIMITER + 0);
//                        System.out.println("Generating bias unit " + id);
                        sendMessage(id, new Text(""));
                    }

                    generateEdgesToNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }

                vertex.voteToHalt();
                break;

            case NumberOfClasses.BACK_EDGES_GENERATION_STATE:
                if (layerNum == MAX_HIDDEN_LAYER_NUM) {
                    generateEdgesFromNextLayer(vertex, networkNum, layerNum, OUTPUT_LAYER,
                            OUTPUT_LAYER_NEURON_COUNT, neuronNum);
                } else if (layerNum == OUTPUT_LAYER) {
                    if (neuronNum == 1) {
                        turnInputLayerToActive(networkNum);

//                        System.out.println("Changing to forward propagation stage");
                        aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                    }

                    vertex.voteToHalt();
                    break;
                } else {
                    generateEdgesFromNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }


                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACK_EDGES_GENERATION_STATE));
                vertex.voteToHalt();
                break;

            case NumberOfClasses.FORWARD_PROPAGATION_STATE:
                // update the weights
                updateFrontWeights(vertex, layerNum, neuronNum);
                updateBackWeights(vertex, layerNum, neuronNum);

                // set new activation
                if (layerNum != INPUT_LAYER && neuronNum != BIAS_UNIT) {
                    double weightedInput = 0d;

                    for (Text m : messages) {
                        Double input = Double.parseDouble(m.toString());
                        weightedInput += input;
//                        System.out.println("message : " + m);
                    }

                    vertex.getValue().setWeightedInput(weightedInput);
                    Double activation = activationFunction(weightedInput);
                    vertex.getValue().setActivation(activation);

//                    System.out.println("Weighted Input: " + vertex.getValue().getWeightedInput());
//                    System.out.println("Activation: " + vertex.getValue().getActivation());
                }

                // aggregate cost
                if (layerNum == OUTPUT_LAYER) {
                    //flush aggregator (only once)
                    if (networkNum == 1 && neuronNum == 1) {
                        DoubleWritable oldCost = getAggregatedValue(NumberOfClasses.COST_AGGREGATOR);
                        aggregate(NumberOfClasses.COST_AGGREGATOR, new DoubleWritable(-oldCost.get()));
                    }

                    double cost = neuronCost(vertex);
//                    System.out.println("Cost frag = " + cost);
                    aggregate(NumberOfClasses.COST_AGGREGATOR, new DoubleWritable(cost));
                }

                // forward propagation
                if (layerNum != NeuralNetworkVertexInputFormat.OUTPUT_LAYER) {
                    //activate bias unit of next layer
                    if (neuronNum == 1 && layerNum != MAX_HIDDEN_LAYER_NUM)
                        sendMessage(new Text(networkNum + DELIMITER + (layerNum + 1) + DELIMITER + BIAS_UNIT), new Text(""));

                    //set correct activation in case of bias unit
                    if (neuronNum == BIAS_UNIT)
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
                if (layerNum != OUTPUT_LAYER) {
                    //flush error aggregator DELTA
                    if (layerNum != OUTPUT_LAYER && networkNum == 1) {
//                        System.out.printf("flushing error agg for layerNum: %s, neuronNum: %d\n", layerNum, neuronNum);
                        int nextLayerNeuronCount = HIDDEN_LAYER_NEURON_COUNT;
                        if (layerNum == MAX_HIDDEN_LAYER_NUM)
                            nextLayerNeuronCount = OUTPUT_LAYER_NEURON_COUNT;

                        // reset error accumulator to zero
                        flushErrorAggregator(layerNum, neuronNum, nextLayerNeuronCount);
                    }

                    double weightedError = 0d;
                    DoubleDenseVector errVector;

                    if (layerNum == MAX_HIDDEN_LAYER_NUM)
                        errVector = new DoubleDenseVector(OUTPUT_LAYER_NEURON_COUNT);
                    else
                        errVector = new DoubleDenseVector(HIDDEN_LAYER_NEURON_COUNT);

                    for (Text m : messages) {
                        String[] msgTokens = m.toString().split(DELIMITER);
                        int senderNeuronNum = Integer.parseInt(msgTokens[0]);
                        Double activation = vertex.getValue().getActivation();
                        Double input = Double.parseDouble(msgTokens[1]);
                        Double srcError = Double.parseDouble(msgTokens[2]);

                        errVector.set(senderNeuronNum - 1, activation * srcError);      // to calculate gradient later
                        weightedError += input;
//                        System.out.println("message   : " + m);
                    }

                    String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
                    aggregate(aggName, errVector);

                    int size;
                    if (layerNum == MAX_HIDDEN_LAYER_NUM)
                        size = OUTPUT_LAYER_NEURON_COUNT;
                    else
                        size = HIDDEN_LAYER_NEURON_COUNT;
//                    System.out.print("errVector : ");
//                    for (int i = 0; i < size; i++) {
//                        System.out.print(errVector.get(i) + "  ");
//                    }
//                    System.out.println("");

                    if (layerNum != INPUT_LAYER) {
                        Double weightedInput = vertex.getValue().getWeightedInput();
                        Double error = weightedError * activationFunctionDerivative(weightedInput);
                        vertex.getValue().setError(error);
//                        System.out.println("Error     : " + error);
                    }
                } else {
                    // output layer error
                    Double error = vertex.getValue().getActivation() - vertex.getValue().getClassFlag();
                    vertex.getValue().setError(error);
//                    System.out.println("New error: " + vertex.getValue().getError());
                }

                // backpropagation of error
                if (layerNum != INPUT_LAYER) {
                    if (neuronNum != BIAS_UNIT) {
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

//        System.out.print("Canceller: ");
        for (int i = 0; i < nextLayerNeuronCount; i++) {
            Double grad = gradients.get(i);
            canceller.set(i, -grad);
//            System.out.print(canceller.get(i) + "  ");
        }
//        System.out.println("");

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

//        System.out.printf("y: %d, activation: %f\n", y, activation);
//        System.out.printf("fragment = %f\n", fragment);
        return fragment;
    }

    private void updateFrontWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        if (layerNum == OUTPUT_LAYER)
            return;

        IntWritable m = getAggregatedValue(NumberOfClasses.NUMBER_OF_NETWORKS_ID);
        String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
        DoubleDenseVector gradients = getAggregatedValue(aggName);

        for (Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
            if (isAnEdgeToNextLayer(e, layerNum)) {
                Text dstId = e.getTargetVertexId();
                String[] edgeTokens = dstId.toString().split(DELIMITER);
                int dstNeuronNum = Integer.parseInt(edgeTokens[2]);

//                System.out.print("gradients: ");
                int nextLayerNeuronCnt = layerNum == MAX_HIDDEN_LAYER_NUM ? OUTPUT_LAYER_NEURON_COUNT : HIDDEN_LAYER_NEURON_COUNT;
                for (int i = 0; i < nextLayerNeuronCnt; i++) {
//                    System.out.print(gradients.get(i) + "  ");
                }
//                System.out.println("");

                Double gradient = gradients.get(dstNeuronNum - 1) / m.get();
                Double old = e.getValue().get();
                Double update = gradient * LEARNING_RATE;

                // gradient descent
                e.getValue().set(old - update);

//                System.out.println("Updating front edge " + vertex.getId() + " --> " + e.getTargetVertexId());
//                System.out.printf("Old val: %f, gradient: %f, update: %f, new val: %f\n",
//                        old, gradient, update, e.getValue().get());
            }
        }
    }

    private void updateBackWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        if (layerNum == INPUT_LAYER)
            return;

        IntWritable m = getAggregatedValue(NumberOfClasses.NUMBER_OF_NETWORKS_ID);
        int prevLayerNum = layerNum == OUTPUT_LAYER ? MAX_HIDDEN_LAYER_NUM : (layerNum - 1);
        int cnt = 0;

        for (Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
            if (isAnEdgeToPrevLayer(e, layerNum)) {
                Text dstId = e.getTargetVertexId();
                String[] tokens = dstId.toString().split(DELIMITER);
                int dstNeuronNum = Integer.parseInt(tokens[2]);

                String aggName = NumberOfClasses.GetErrorAggregatorName(prevLayerNum, dstNeuronNum);
                DoubleDenseVector gradients = getAggregatedValue(aggName);

                Double gradient = gradients.get(neuronNum - 1) / m.get();
                Double old = e.getValue().get();
                Double update = gradient * LEARNING_RATE;

//                System.out.println("Updating back edge " + vertex.getId() + " --> " + e.getTargetVertexId());
//                System.out.printf("Old val: %f, gradient: %f, update: %f, new val: %f\n",
//                        old, gradient, update, (old - update));

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
                String msg = neuronNum + DELIMITER + fragment + DELIMITER + error;
                sendMessage(dstId, new Text(msg));
//                System.out.printf("weight: %f, error: %f\n", weight, error);
//                System.out.println("Sending msg to " + dstId + ", msg = " + msg);
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
//                System.out.printf("weight: %f, activation: %f\n", weight, activation);
//                System.out.println("Sending msg to " + dstId + ", msg = " + fragment);
            }
        }
    }

    private boolean isAnEdgeToNextLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[1]);

        if (dstLayerNum == OUTPUT_LAYER) {
            return layerNum == MAX_HIDDEN_LAYER_NUM;
        } else {
            return (layerNum + 1 == dstLayerNum);
        }
    }

    private boolean isAnEdgeToPrevLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[1]);

        if (dstLayerNum == MAX_HIDDEN_LAYER_NUM) {
            return layerNum == OUTPUT_LAYER;
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
//            System.out.println("Generated edge from " + vertex.getId() + " to " +
//                    e.getTargetVertexId() + " with weight " + e.getValue());
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
//            System.out.println("Generated edge from " + srcId + " to " +
//                    e.getTargetVertexId() + " with weight " + e.getValue());

            sendMessage(srcId, new Text(""));       // to activate previous layer
        }

        //activate bias unit of next layer
        Text biasDst = new Text(String.format("%d%s%d%s%d", networkNum, DELIMITER, nextLayer, DELIMITER, BIAS_UNIT));
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
        for (int i = 0; i <= INPUT_LAYER_NEURON_COUNT; i++) {
            Text dstId = new Text(String.format("%d:%d:%d", networkNum, INPUT_LAYER, i));
            sendMessage(dstId, new Text(""));
        }
    }
}

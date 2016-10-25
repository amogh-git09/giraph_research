package neural_net;

import org.apache.commons.math3.analysis.function.Sigmoid;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVector;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class BackwardPropagation extends
        BasicComputation<Text, NeuronValue, DoubleWritable, Text> {

    static final int MAX_HIDDEN_LAYER_NUM = 3;                 // minimum value 2
    static final int HIDDEN_LAYER_NEURON_COUNT = 4;
    static final int INPUT_LAYER_NEURON_COUNT = 4;
    static final int OUTPUT_LAYER_NEURON_COUNT = 1;
    static final int FIRST_HIDDEN_LAYER = 2;
    static final int INPUT_LAYER = 1;
    static final int OUTPUT_LAYER = -1;
    static final String DELIMITER = ":";

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        System.out.println("SS: " + getSuperstep() + "  Vertex ID: " + vertex.getId());

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

        IntWritable state = getAggregatedValue(NumberOfClasses.STATE_ID);

        switch (state.get()) {
            case NumberOfClasses.HIDDEN_LAYER_GENERATION_STATE:
                if (layerNum == MAX_HIDDEN_LAYER_NUM) {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, OUTPUT_LAYER,
                            OUTPUT_LAYER_NEURON_COUNT, neuronNum);
                } else if(layerNum == OUTPUT_LAYER && neuronNum == 1) {
                    turnInputLayerToActive(networkNum);
                    System.out.println("Setting to BACK EDGES GENERATION");
                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACK_EDGES_GENERATION_STATE));
                } else {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }
                vertex.voteToHalt();
                break;

            case NumberOfClasses.BACK_EDGES_GENERATION_STATE:
                if(layerNum == MAX_HIDDEN_LAYER_NUM) {
                    generateEdgesFromNextLayer(vertex, networkNum, layerNum, OUTPUT_LAYER,
                            OUTPUT_LAYER_NEURON_COUNT, neuronNum);
                } else if(layerNum == OUTPUT_LAYER && neuronNum == 1) {
                    turnInputLayerToActive(networkNum);
                    System.out.println("Setting to FORWARD PROPAGATION");
                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
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


                if(layerNum != INPUT_LAYER) {
                    double weightedInput = 0d;

                    for(Text m : messages) {
                        Double input = Double.parseDouble(m.toString());
                        weightedInput += input;
                        System.out.println("message : " + m);
                    }

                    vertex.getValue().setWeightedInput(weightedInput);
                    Double activation = activationFunction(weightedInput);
                    vertex.getValue().setActivation(activation);
                    System.out.println("Weighted Input: " + vertex.getValue().getWeightedInput());
                    System.out.println("Activation: " + vertex.getValue().getActivation());
                }

                if(layerNum != NeuralNetworkVertexInputFormat.OUTPUT_LAYER)
                    forwardProp(vertex, layerNum);
                else {
                    Double error = vertex.getValue().getActivation() - vertex.getValue().getError();
                    vertex.getValue().setError(error);
                    System.out.println("New error: " + vertex.getValue().getError());
                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACKWARD_PROPAGATION_STATE));
                    break;
                }

                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                vertex.voteToHalt();
                break;

            case NumberOfClasses.BACKWARD_PROPAGATION_STATE:
                if(layerNum != OUTPUT_LAYER) {
                    double weightedError = 0d;
                    DoubleDenseVector errVector;

                    if(layerNum == MAX_HIDDEN_LAYER_NUM)
                        errVector = new DoubleDenseVector(OUTPUT_LAYER_NEURON_COUNT);
                    else
                        errVector = new DoubleDenseVector(HIDDEN_LAYER_NEURON_COUNT);

                    for(Text m: messages) {
                        String[] msgTokens = m.toString().split(DELIMITER);
                        int senderNeuronNum = Integer.parseInt(msgTokens[0]);
                        Double input = Double.parseDouble(msgTokens[1]);

                        errVector.set(senderNeuronNum - 1, input);      // to calculate gradient later
                        weightedError += input;
                        System.out.println("message   : " + m);
                    }

                    String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
                    aggregate(aggName, errVector);

                    int size;
                    if(layerNum == MAX_HIDDEN_LAYER_NUM)
                        size = OUTPUT_LAYER_NEURON_COUNT;
                    else
                        size = HIDDEN_LAYER_NEURON_COUNT;
                    System.out.print("errVector : ");
                    for(int i=0; i<size; i++) {
                        System.out.print(errVector.get(i) + "  ");
                    }
                    System.out.println("");

                    if(layerNum != INPUT_LAYER) {
                        Double weightedInput = vertex.getValue().getWeightedInput();
                        Double error = weightedError*activationFunctionDerivative(weightedInput);
                        vertex.getValue().setError(error);
                        System.out.println("Error     : " + error);
                    }
                }

                if(layerNum != INPUT_LAYER) {
                    backPropagateError(vertex, layerNum, neuronNum);
                }

                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACKWARD_PROPAGATION_STATE));
                vertex.voteToHalt();
                break;
        }
    }

    private void updateWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        
    }

    private void backPropagateError(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        for (Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            Text dstId = e.getTargetVertexId();

            if (isAnEdgeToPrevLayer(e, layerNum)) {
                DoubleWritable weight = e.getValue();
                Double error = vertex.getValue().getError();
                Double fragment = weight.get()*error;
                sendMessage(dstId, new Text(neuronNum + DELIMITER + fragment));
                System.out.println("Sending msg to " + dstId + ", " + fragment);
            }
        }
    }

    private void forwardProp(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum) {
        for(Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            Text dstId = e.getTargetVertexId();

            if(isAnEdgeToNextLayer(e, layerNum)) {
                DoubleWritable weight = e.getValue();
                Double activation = vertex.getValue().getActivation();
                Double fragment = weight.get()*activation;
                sendMessage(dstId, new Text(fragment + ""));
                System.out.println("Sending msg to " + dstId);
            }
        }
    }

    private boolean isAnEdgeToNextLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[1]);

        if(dstLayerNum == OUTPUT_LAYER) {
            return layerNum == MAX_HIDDEN_LAYER_NUM;
        } else {
            return (layerNum + 1 == dstLayerNum);
        }
    }

    private boolean isAnEdgeToPrevLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[1]);

        if(dstLayerNum == MAX_HIDDEN_LAYER_NUM) {
            return layerNum == OUTPUT_LAYER;
        } else {
            return (layerNum - 1 == dstLayerNum);
        }
    }

    private void generateEdgesToNextLayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                          int networkNum, int thisLayer, int nextLayer,
                                          int nextLayerCount, int neuronNum) throws IOException {

        for (int i = 1; i <= nextLayerCount; i++) {
            DoubleDenseVector weights = getAggregatedValue(
                    NumberOfClasses.GetWeightAggregatorName(thisLayer, neuronNum));
            Double weight = weights.get(i-1);

            Text dstId = new Text(networkNum + ":" + nextLayer + ":" + i);
            sendMessage(dstId, new Text(""));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(weight));
            addEdgeRequest(vertex.getId(), e);
            System.out.println("Generated edge from " + vertex.getId() + " to " +
                    e.getTargetVertexId() + " with weight " + e.getValue());
        }
    }

    private void generateEdgesFromNextLayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                          int networkNum, int thisLayer, int nextLayer,
                                          int nextLayerCount, int neuronNum) throws IOException {

        for (int i = 1; i <= nextLayerCount; i++) {
            DoubleDenseVector weights = getAggregatedValue(
                    NumberOfClasses.GetWeightAggregatorName(thisLayer, neuronNum));
            Double weight = weights.get(i-1);

            Text srcId = new Text(networkNum + ":" + nextLayer + ":" + i);
            Edge<Text, DoubleWritable> e = EdgeFactory.create(vertex.getId(), new DoubleWritable(weight));
            addEdgeRequest(srcId, e);
            System.out.println("Generated edge from " + srcId + " to " +
                    e.getTargetVertexId() + " with weight " + e.getValue());

            sendMessage(srcId, new Text(""));       // to activate previous layer
        }
    }

    // input x should the weighted input
    private double activationFunction(Double x) {
        Sigmoid sig = new Sigmoid();
        return sig.value(x);
    }

    // input x must be the weighted input
    private double activationFunctionDerivative(Double x) {
        Sigmoid sig = new Sigmoid();
        return sig.value(x)*(1 - sig.value(x));
    }

    private void turnInputLayerToActive(int networkNum) {
        for(int i = 1; i <= INPUT_LAYER_NEURON_COUNT; i++) {
            Text dstId = new Text(String.format("%d:%d:%d", networkNum, INPUT_LAYER, i));
            sendMessage(dstId, new Text(""));
        }
    }
}

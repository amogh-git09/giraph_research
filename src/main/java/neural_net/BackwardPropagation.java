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
        BasicComputation<Text, NeuronValue, DoubleWritable, DoubleWritable> {

    static final int MAX_HIDDEN_LAYER_NUM = 3;                 // minimum value 2
    static final int HIDDEN_LAYER_NEURON_COUNT = 4;
    static final int INPUT_LAYER_NEURON_COUNT = 4;
    static final int OUTPUT_LAYER_NEURON_COUNT = 1;
    static final int OUTPUT_LAYER = -1;
    static final String DELIMITER = ":";

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<DoubleWritable> messages) throws IOException {

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
                    for(int i = 1; i <= INPUT_LAYER_NEURON_COUNT; i++) {
                        Text dstId = new Text(String.format("%d:%d:%d", networkNum,
                                NeuralNetworkVertexInputFormat.INPUT_LAYER, i));
                        sendMessage(dstId, new DoubleWritable(0));
                    }

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
                    for(int i = 1; i <= INPUT_LAYER_NEURON_COUNT; i++) {
                        Text dstId = new Text(String.format("%d:%d:%d", networkNum,
                                NeuralNetworkVertexInputFormat.INPUT_LAYER, i));
                        sendMessage(dstId, new DoubleWritable(0));
                    }
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
                if(layerNum != NeuralNetworkVertexInputFormat.INPUT_LAYER) {
                    Double activation = 0d;

                    for(DoubleWritable m : messages) {
                        activation += m.get();
                        System.out.println("message : " + m);
                    }

                    activation = activationFunction(activation);

                    vertex.getValue().setActivation(activation);
//                    System.out.println("New activation: " + vertex.getValue().getActivation());
//                    System.out.println("Error: " + vertex.getValue().getError());
                }

                if(layerNum != NeuralNetworkVertexInputFormat.OUTPUT_LAYER)
                    forwardProp(vertex);
                else {
                    Double error = vertex.getValue().getActivation() - vertex.getValue().getError();
                    vertex.getValue().setError(error);
//                    System.out.println("New Error: " + vertex.getValue().getError());

                    for(int i=1; i<=HIDDEN_LAYER_NEURON_COUNT; i++) {

                    }

                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.BACKWARD_PROPAGATION_STATE));
                }

                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                vertex.voteToHalt();
                break;
        }
    }

    private void forwardProp(Vertex<Text, NeuronValue, DoubleWritable> vertex) {
        for(Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            Text dstId = e.getTargetVertexId();
            String[] edgeTokens = dstId.toString().split(DELIMITER);
            String[] vertexTokens = vertex.getId().toString().split(DELIMITER);

            int dstLayerNum = Integer.parseInt(edgeTokens[1]);
            int vertexLayerNum = Integer.parseInt(vertexTokens[1]);

            if((vertexLayerNum + 1 == dstLayerNum) || (dstLayerNum == OUTPUT_LAYER)) {
                DoubleWritable weight = e.getValue();
                Double activation = vertex.getValue().getActivation();
                Double fragment = weight.get()*activation;
                sendMessage(dstId, new DoubleWritable(fragment));
                System.out.println("Sending msg to " + dstId);
            }
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
            sendMessage(dstId, new DoubleWritable(0));                   // adds a new vertex if not existent
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

            sendMessage(srcId, new DoubleWritable(0));
        }
    }

    private double activationFunction(Double x) {
        Sigmoid sig = new Sigmoid();
        return sig.value(x);
    }
}

package neural_net;

import org.apache.giraph.aggregators.AggregatorUsage;
import org.apache.giraph.aggregators.matrix.dense.IntDenseMatrix;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVector;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;
import java.util.HashMap;
import java.util.Random;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class BackwardPropagation extends
        BasicComputation<Text, DoubleWritable, DoubleWritable, Text> {

    static final int MAX_HIDDEN_LAYER_NUM = 3;                 // minimum value 2
    static final int HIDDEN_LAYER_NEURON_COUNT = 4;
    static final int OUTPUT_LAYER = -1;
    static final double EPSILON = 0.2;
    static final Random random = new Random();
//    HashMap<String, Double> weights = new HashMap<String, Double>();

    @Override
    public void compute(Vertex<Text, DoubleWritable, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        System.out.println("SS: " + getSuperstep() + "  Vertex NUMBER_OF_CLASSES_ID: " + vertex.getId());

        String[] tokens = vertex.getId().toString().split(":");
        int networkNum = Integer.parseInt(tokens[0]);
        int layerNum = Integer.parseInt(tokens[1]);
        int neuronNum = Integer.parseInt(tokens[2]);

        if (getSuperstep() == 0) {
            if(networkNum == 1) {
                //aggregate number of hidden layers
                aggregate(NumberOfClasses.NUMBER_OF_HIDDEN_LAYERS_ID, new IntWritable(HIDDEN_LAYER_NEURON_COUNT));

                //aggregate the number of neurons in a hidden layer
                aggregate(NumberOfClasses.HIDDEN_LAYER_NEURON_COUNT, new IntWritable(HIDDEN_LAYER_NEURON_COUNT));
            }

            if (layerNum == NeuralNetworkVertexInputFormat.OUTPUT_LAYER) {
                if (networkNum == 1) {
                    aggregate(NumberOfClasses.NUMBER_OF_CLASSES_ID, new IntWritable(1));
                }
                //aggregate number of networks
                aggregate(NumberOfClasses.NUMBER_OF_NETWORKS_ID, new IntWritable(networkNum));
                vertex.voteToHalt();
                return;
            } else if (layerNum == NeuralNetworkVertexInputFormat.INPUT_LAYER) {
                if (networkNum == 1) {
                   aggregate(NumberOfClasses.NUMBER_OF_INPUT_NEUTRONS_ID, new IntWritable(1));
                }
            }
        }

        IntWritable state = getAggregatedValue(NumberOfClasses.STATE_ID);

        switch (state.get()) {
            case NumberOfClasses.HIDDEN_LAYER_GENERATION:
                if (layerNum == MAX_HIDDEN_LAYER_NUM) {
                    IntWritable numClasses = getAggregatedValue(NumberOfClasses.NUMBER_OF_CLASSES_ID);
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, OUTPUT_LAYER,
                            numClasses.get(), neuronNum);
                } else if(layerNum == OUTPUT_LAYER && neuronNum == 1) {
                    IntWritable numInputNeurons = getAggregatedValue(NumberOfClasses.NUMBER_OF_INPUT_NEUTRONS_ID);
                    IntWritable numNetworks = getAggregatedValue(NumberOfClasses.NUMBER_OF_NETWORKS_ID);
//                    for(int i = 1; i <= numInputNeurons.get(); i++) {
//                        Text dstId = new Text(String.format("%d:%d:%d", networkNum,
//                                NeuralNetworkVertexInputFormat.INPUT_LAYER, i));
//                        sendMessage(dstId, new Text(""));
//                    }
                } else {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }
                vertex.voteToHalt();
                break;

//            case FORWARD_PROPAGATION:
//                if(layerNum != NeuralNetworkVertexInputFormat.INPUT_LAYER) {
//                    for(Text m : messages) {
//                        System.out.println("message : " + m);
//                    }
//                }
//
//                if(layerNum != NeuralNetworkVertexInputFormat.OUTPUT_LAYER)
//                    forwardProp(vertex);
//                vertex.voteToHalt();
//                break;
        }
    }

    private void forwardProp(Vertex<Text, DoubleWritable, DoubleWritable> vertex) {
        for(Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            Text dstId = e.getTargetVertexId();
            DoubleWritable weight = e.getValue();
            DoubleWritable activation = vertex.getValue();
            Double fragment = weight.get()*activation.get();
            Text msg = new Text("" + (new DoubleWritable(fragment)));
            sendMessage(dstId, msg);
        }
    }

    private void generateEdgesToNextLayer(Vertex<Text, DoubleWritable, DoubleWritable> vertex, int networkNum, int srcLayer, int dstLayer,
                                          int nextLayerCount, int neuronNum) throws IOException {

        for (int i = 1; i <= nextLayerCount; i++) {
            Double randWeight;
            String weightId = srcLayer + ":" + neuronNum + ":" + dstLayer + ":" + i;

            randWeight = getRandomInRange(-EPSILON, EPSILON);

            Text dstId = new Text(networkNum + ":" + dstLayer + ":" + i);
            weights.put(weightId, randWeight);

            sendMessage(dstId, new Text(""));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(randWeight));
            addEdgeRequest(vertex.getId(), e);
//            System.out.println("Generated edge from " + vertex.getId() + " to " +
//                    e.getTargetVertexId() + " with weight " + e.getValue());
        }
    }

    public IntDenseMatrix getMatrix(int numRows, AggregatorUsage aggUser) {
        IntDenseMatrix matrix = new IntDenseMatrix(numRows, 1);
        for (int i = 0; i < numRows; ++i) {
            IntDenseVector vec = aggUser.getAggregatedValue(getRowAggregatorName(i));
            matrix.addRow(vec);
        }
        return matrix;
    }
}

package neural_net;

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

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<DoubleWritable> messages) throws IOException {

        System.out.println("SS: " + getSuperstep() + "  Vertex ID: " + vertex.getId());

        String[] tokens = vertex.getId().toString().split(":");
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

                    System.out.println("Setting to FORWARD PROPAGATION");
                    aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                } else {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }
                vertex.voteToHalt();
                break;

            case NumberOfClasses.FORWARD_PROPAGATION_STATE:
                if(layerNum != NeuralNetworkVertexInputFormat.INPUT_LAYER) {
                    Double activation = 0d;

                    for(DoubleWritable m : messages) {
                        activation += m.get();
                        System.out.println("message : " + m);
                    }

                    vertex.getValue().setActivation(activation);

                    System.out.println("New activation: " + vertex.getValue().getActivation());
                    System.out.println("Error: " + vertex.getValue().getError());
                }

                if(layerNum != NeuralNetworkVertexInputFormat.OUTPUT_LAYER)
                    forwardProp(vertex);

                aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                vertex.voteToHalt();
                break;
        }
    }

    private void forwardProp(Vertex<Text, NeuronValue, DoubleWritable> vertex) {
        for(Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            Text dstId = e.getTargetVertexId();
            DoubleWritable weight = e.getValue();
            Double activation = vertex.getValue().getActivation();
            Double fragment = weight.get()*activation;
            sendMessage(dstId, new DoubleWritable(fragment));
        }
    }

    private void generateEdgesToNextLayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                          int networkNum, int srcLayer, int dstLayer,
                                          int nextLayerCount, int neuronNum) throws IOException {

        for (int i = 1; i <= nextLayerCount; i++) {
            DoubleDenseVector weights = getAggregatedValue(
                    NumberOfClasses.GetWeightAggregatorName(srcLayer, neuronNum));
            Double weight = weights.get(i-1);

            Text dstId = new Text(networkNum + ":" + dstLayer + ":" + i);
            sendMessage(dstId, new DoubleWritable(0));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(weight));
            addEdgeRequest(vertex.getId(), e);
            System.out.println("Generated edge from " + vertex.getId() + " to " +
                    e.getTargetVertexId() + " with weight " + e.getValue());
        }
    }

}

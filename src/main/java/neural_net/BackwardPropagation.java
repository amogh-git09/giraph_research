package neural_net;

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

    enum State {
        HIDDEN_LAYER_GENERATION
    }

    final int MAX_HIDDEN_LAYER_NUM = 3;                 // minimum value 2
    final int HIDDEN_LAYER_NEURON_COUNT = 4;
    final int OUTPUT_LAYER = -1;
    final double EPSILON = 0.2;
    int workingLayer = 1;
    State state = State.HIDDEN_LAYER_GENERATION;
    final Random random = new Random();
    HashMap<String, Double> weights = new HashMap<String, Double>();

    @Override
    public void compute(Vertex<Text, DoubleWritable, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        System.out.println("SS: " + getSuperstep() + "  Vertex ID: " + vertex.getId());
        String[] tokens = vertex.getId().toString().split(":");
        int networkNum = Integer.parseInt(tokens[0]);
        int layerNum = Integer.parseInt(tokens[1]);
        int neuronNum = Integer.parseInt(tokens[2]);

        if (getSuperstep() == 0 && layerNum == -1) {
            if (networkNum == 1) {
                aggregate(NumberOfClasses.ID, new IntWritable(1));
            }
            vertex.voteToHalt();
            return;
        }

        switch (state) {
            case HIDDEN_LAYER_GENERATION:
                if (layerNum == MAX_HIDDEN_LAYER_NUM) {
                    IntWritable numClasses = getAggregatedValue(NumberOfClasses.ID);
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, OUTPUT_LAYER,
                            numClasses.get(), neuronNum);
                } else if(layerNum == OUTPUT_LAYER) {
                    vertex.voteToHalt();
                } else {
                    generateEdgesToNextLayer(vertex, networkNum, layerNum, layerNum + 1,
                            HIDDEN_LAYER_NEURON_COUNT, neuronNum);
                }
        }
    }

    private void generateEdgesToNextLayer(Vertex<Text, DoubleWritable, DoubleWritable> vertex, int networkNum, int srcLayer, int dstLayer,
                                          int nextLayerCount, int neuronNum) throws IOException {

        for (int i = 1; i <= nextLayerCount; i++) {
            Double randWeight;
            String weightId = srcLayer + ":" + neuronNum + ":" + dstLayer + ":" + i;

            if (weights.containsKey(weightId)) {
                randWeight = weights.get(weightId);
            } else {
                randWeight = getRandomInRange(-EPSILON, EPSILON);
            }

            Text dstId = new Text(networkNum + ":" + dstLayer + ":" + i);
            weights.put(weightId, randWeight);

            sendMessage(dstId, new Text(""));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(randWeight));
            addEdgeRequest(vertex.getId(), e);
            System.out.println("Generated edge from " + vertex.getId() + " to " +
                    e.getTargetVertexId() + " with weight " + e.getValue());
        }

        vertex.voteToHalt();
    }

    private double getRandomInRange(Double min, Double max) {
        Double rand = random.nextDouble();
        return min + (max - min) * rand;
    }
}

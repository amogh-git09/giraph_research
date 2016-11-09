package distributed_net;

import debug.Logger;
import master_compute.NNMasterCompute;
import org.apache.giraph.GiraphRunner;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ToolRunner;
import worker_context.RedisWorkerContext;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/11/09.
 */
public class DistributedNeuralNetwork extends
        BasicComputation<Text, NeuronValue, DoubleWritable, Text> {

    private RedisWorkerContext workerContext;

    public static final int INPUT_LAYER_NEURON_COUNT = 4;
    public static final int INPUT_LAYER = 1;
    public static final int OUTPUT_LAYER = -1;
    public static final int MAX_HIDDEN_LAYER_NUM = 2;
    public static final int BIAS_UNIT = 0;
    public static final int HIDDEN_LAYER_NEURON_COUNT = 2;
    public static final int OUTPUT_LAYER_NEURON_COUNT = 1;

    public final String DELIMITER = ":";

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new GiraphRunner(), args));
    }

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        if (workerContext == null) {
            workerContext = (RedisWorkerContext) getWorkerContext();
        }

        //generate input layer neurons
        if(getSuperstep() == 0) {
            for(int i = 1; i <= INPUT_LAYER_NEURON_COUNT; i++) {
                Text destVertexID = new Text(getNeuronId(INPUT_LAYER, i));
                NeuronValue nVal = new NeuronValue();
                addVertexRequest(destVertexID, nVal);
            }

            //switch to forward edges generation stage
            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
            return;
        }

        IntWritable stage = getAggregatedValue(NNMasterCompute.STAGE_AGG_ID);
        String[] tokens = vertex.getId().toString().split(DELIMITER);
        int layerNum = Integer.parseInt(tokens[0]);
        int neuronNum = Integer.parseInt(tokens[1]);

        Logger.i(String.format("SS: %d, vertexId: %s, Stage: %d", getSuperstep(), vertex.getId(), stage.get()));

        switch (stage.get()) {
            case NNMasterCompute.FRONT_EDGES_GENERATION_STAGE:
                switch (layerNum) {
                    //generate the output layer
                    case MAX_HIDDEN_LAYER_NUM:
                        generateEdgesToNextLayer(vertex, OUTPUT_LAYER, OUTPUT_LAYER_NEURON_COUNT);
                        vertex.voteToHalt();
                        break;

                    //switch to back edges generation stage
                    case OUTPUT_LAYER:
                        if (neuronNum == 1) {
                            Logger.d("Switching to next stage");
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
                            turnInputLayerToActive();
                        }
                        vertex.voteToHalt();
                        break;

                    default:
                        //generate next layer's bias unit
                        if (neuronNum == BIAS_UNIT) {
                            Text id = new Text(getNeuronId(layerNum + 1, BIAS_UNIT));
                            sendMessage(id, new Text(""));
                        }

                        // generate the next hidden layer
                        generateEdgesToNextLayer(vertex, layerNum + 1, HIDDEN_LAYER_NEURON_COUNT);
                        vertex.voteToHalt();
                }
                break;

            default:
                vertex.voteToHalt();
        }
    }

    public String getNeuronId(int layerNum, int neuronNum) {
        return layerNum + DELIMITER + neuronNum;
    }

    private void generateEdgesToNextLayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                          int targetLayerNum, int nextLayerCount) throws IOException {

        // no edges to bias unit
        for (int i = 1; i <= nextLayerCount; i++) {
            Double weight = workerContext.getRandomWeight();

            Text dstId = new Text(getNeuronId(targetLayerNum, i));
            sendMessage(dstId, new Text(""));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(weight));
            addEdgeRequest(vertex.getId(), e);
            Logger.d(String.format("Generated edge from %s to %s with weight %f", vertex.getId().toString(),
                    e.getTargetVertexId().toString(), e.getValue().get()));
        }
    }

    private void turnInputLayerToActive() {
        for (int i = 0; i <= INPUT_LAYER_NEURON_COUNT; i++) {
            Text dstId = new Text(getNeuronId(INPUT_LAYER, i));
            sendMessage(dstId, new Text(""));
        }
    }
}

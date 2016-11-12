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
    public int dataSetNum = 1;

    public static final int INPUT_LAYER_NEURON_COUNT = 4;
    public static final int INPUT_LAYER = 1;
    public static final int OUTPUT_LAYER = -1;
    public static final int MAX_HIDDEN_LAYER_NUM = 2;
    public static final int BIAS_UNIT = 0;
    public static final int OUTPUT_LAYER_NEURON_COUNT = 1;
    private static final int[] LAYER_TO_NEURON = {
            INPUT_LAYER_NEURON_COUNT, 2, OUTPUT_LAYER_NEURON_COUNT};

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
                        generateEdgesToALayer(vertex, getNextLayerNum(layerNum), false);
                        vertex.voteToHalt();
                        break;

                    //switch to back edges generation stage
                    case OUTPUT_LAYER:
                        if (neuronNum == 1) {
                            Logger.d("Switching to next stage");
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
                            activateLayer(INPUT_LAYER, true);
                            vertex.voteToHalt();
                        }
                        break;

                    default:
                        //generate next layer's bias unit
                        if (neuronNum == BIAS_UNIT) {
                            Text id = new Text(getNeuronId(getNextLayerNum(layerNum), BIAS_UNIT));
                            sendMessage(id, new Text(""));
                        }

                        // generate the next hidden layer
                        generateEdgesToALayer(vertex, getNextLayerNum(layerNum), false);
                        vertex.voteToHalt();
                }
                break;

            case NNMasterCompute.BACK_EDGES_GENERATION_STAGE:
                switch (layerNum) {
                    case OUTPUT_LAYER:
                        //switch to next stage
                        if(neuronNum == 1) {
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
                        }
                        break;

                    default:
                        //generate back edges
                        generateBackEdges(vertex, layerNum);
                        int nextLayer = getNextLayerNum(layerNum);
                        activateLayer(nextLayer, containsBiasUnit(nextLayer));
                        vertex.voteToHalt();
                        break;
                }
                break;

            case NNMasterCompute.DATA_LOAD_STAGE:
                switch (layerNum) {
                    case INPUT_LAYER:
                        if(neuronNum != BIAS_UNIT) {
                            double input = workerContext.getInputData(dataSetNum, neuronNum);
                            vertex.getValue().setActivation(input);
                        }
                        Logger.d("Loaded activation: " + vertex.getValue().getActivation());
                        vertex.voteToHalt();
                        break;

                    case OUTPUT_LAYER:
                        int classFlag = workerContext.getOutputData(dataSetNum, neuronNum);
                        vertex.getValue().setClassFlag(classFlag);
                        Logger.d("Loaded classFlag: " + vertex.getValue().getClassFlag());
                        activateLayer(INPUT_LAYER, true);
                        vertex.voteToHalt();
                        break;
                }
                break;

            default:
                vertex.voteToHalt();
        }
    }

    public String getNeuronId(int layerNum, int neuronNum) {
        return layerNum + DELIMITER + neuronNum;
    }

    private void generateEdgesToALayer(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                       int targetLayerNum, boolean includeBiasUnit) throws IOException {

        int targetLayerNeuronCount = getNeuronCount(targetLayerNum);
        int i = includeBiasUnit ? 0 : 1;

        for (; i <= targetLayerNeuronCount; i++) {
            Double weight = workerContext.getRandomWeight();
            Text dstId = new Text(getNeuronId(targetLayerNum, i));
            Logger.d("Generating vertex: " + dstId);
            sendMessage(dstId, new Text(""));                   // adds a new vertex if not existent
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(weight));
            addEdgeRequest(vertex.getId(), e);
            Logger.d(String.format("Generated edge from %s to %s with weight %f", vertex.getId().toString(),
                    e.getTargetVertexId().toString(), e.getValue().get()));
        }
    }

    private void generateBackEdges(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum) throws IOException {
        Logger.d("Num Edges = " + vertex.getNumEdges());
        for (Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            if(!isAnEdgeToNextLayer(e, layerNum))
                continue;

            Text srcId = new Text(e.getTargetVertexId());
            DoubleWritable weight = e.getValue();

            Edge<Text, DoubleWritable> backEdge = EdgeFactory.create(vertex.getId(), weight);
            addEdgeRequest(srcId, backEdge);
            Logger.d(String.format("Generated back edge from %s to %s with weight %f", srcId.toString(),
                    backEdge.getTargetVertexId().toString(), backEdge.getValue().get()));
        }
    }

    public int getNeuronCount(int layerNum) {
        if(layerNum == OUTPUT_LAYER)
            return LAYER_TO_NEURON[LAYER_TO_NEURON.length-1];
        else
            return LAYER_TO_NEURON[layerNum - 1];
    }

    private void activateLayer(int layerNum, boolean biasUnitExists) {
        int i = biasUnitExists ? 0 : 1;
        for (; i <= getNeuronCount(layerNum); i++) {
            Text dstId = new Text(getNeuronId(layerNum, i));
            Logger.d("Activating " + dstId);
            sendMessage(dstId, new Text(""));
        }
    }

    private int getNextLayerNum(int layerNum) {
        switch (layerNum) {
            case MAX_HIDDEN_LAYER_NUM:
                return OUTPUT_LAYER;
            case OUTPUT_LAYER:
                return INPUT_LAYER;
            default:
                return layerNum + 1;
        }
    }

    private boolean isAnEdgeToNextLayer(Edge<Text, DoubleWritable> e, int layerNum) {
        Text dstId = e.getTargetVertexId();
        String[] edgeTokens = dstId.toString().split(DELIMITER);
        int dstLayerNum = Integer.parseInt(edgeTokens[0]);
        return dstLayerNum == getNextLayerNum(layerNum);
    }

    private boolean containsBiasUnit(int layerNum) {
        return layerNum != OUTPUT_LAYER;
    }
}

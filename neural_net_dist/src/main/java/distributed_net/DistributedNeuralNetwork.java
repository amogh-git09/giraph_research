package distributed_net;

import debug.Logger;
import master_compute.NNMasterCompute;
import org.apache.commons.math3.analysis.function.Sigmoid;
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
    public static final int DATA_SIZE = 2;
    public static final double LEARNING_RATE = 0.1;

    public static final int INPUT_LAYER_NEURON_COUNT = 4;
    public static final int INPUT_LAYER = 1;
    public static final int OUTPUT_LAYER = -1;
    public static final int MAX_HIDDEN_LAYER_NUM = 2;
    public static final int FIRST_HIDDEN_LAYER_NUM = 2;
    public static final int BIAS_UNIT = 0;
    public static final int OUTPUT_LAYER_NEURON_COUNT = 1;
    private static final int[] LAYER_TO_NEURON = {
            INPUT_LAYER_NEURON_COUNT, 2, OUTPUT_LAYER_NEURON_COUNT};

    public static final String DELIMITER = ":";

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new GiraphRunner(), args));
    }

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        if (workerContext == null) {
            workerContext = (RedisWorkerContext) getWorkerContext();
        }

        IntWritable stage = getAggregatedValue(NNMasterCompute.STAGE_AGG_ID);
        String[] tokens = vertex.getId().toString().split(DELIMITER);
        int layerNum = Integer.parseInt(tokens[0]);
        int neuronNum = Integer.parseInt(tokens[1]);
        IntWritable dataSetIndex = getAggregatedValue(NNMasterCompute.DATA_SET_INDEX_AGG);

        //generate network
        if (getSuperstep() == 0) {
            generateNetwork();
            //switch to stabilizing stage
            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
            return;
        }

        Logger.i(String.format("SS: %d, vertexId: %s, Stage: %s", getSuperstep(),
                vertex.getId(), NNMasterCompute.idToStage(stage.get())));

        switch (stage.get()) {
            case NNMasterCompute.STABILIZE_INITIAL_NETWORK:
                switch (layerNum) {
                    case INPUT_LAYER:
                        //switch to next stage
                        if(neuronNum == 1)
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
                        break;

                    default:
                        vertex.voteToHalt();
                }
                break;

            case NNMasterCompute.FRONT_EDGES_GENERATION_STAGE:
                switch (layerNum) {
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
                        if (neuronNum == BIAS_UNIT && layerNum != MAX_HIDDEN_LAYER_NUM) {
                            Text id = new Text(getNeuronId(getNextLayerNum(layerNum), BIAS_UNIT));
                            Logger.d("Generating vertex: " + id);
                            addVertexRequest(id, new NeuronValue(1d, 0d, 0d, 0, getNeuronCount(getNextLayerNum(layerNum))));
                        }

                        // generate the next hidden layer
                        generateEdgesToALayer(vertex, getNextLayerNum(layerNum), false);
                        // activate next layer
                        if(neuronNum == 1)
                            activateLayer(getNextLayerNum(layerNum), true);
                        vertex.voteToHalt();
                }
                break;

            case NNMasterCompute.BACK_EDGES_GENERATION_STAGE:
                switch (layerNum) {
                    case OUTPUT_LAYER:
                        //switch to next stage
                        if (neuronNum == 1) {
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
                            //activate input layer
                            activateLayer(INPUT_LAYER, true);
                        }
                        break;

                    default:
                        //generate back edges
                        generateBackEdges(vertex, layerNum);

                        //activate next layer
                        if(neuronNum == 1) {
                            int nextLayer = getNextLayerNum(layerNum);
                            activateLayer(nextLayer, containsBiasUnit(nextLayer));
                        }

                        vertex.voteToHalt();
                        break;
                }
                break;

            case NNMasterCompute.DATA_LOAD_STAGE:
                switch (layerNum) {
                    case INPUT_LAYER:
                        if (neuronNum != BIAS_UNIT) {
                            double input = workerContext.getInputData(dataSetIndex.get(), neuronNum);
                            vertex.getValue().setActivation(input);
                            Logger.d("Loaded activation: " + vertex.getValue().getActivation());
                        }

                        //switch stage
                        if (neuronNum == 1)
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));

                        break;

                    case OUTPUT_LAYER:
                        int classFlag = workerContext.getOutputData(dataSetIndex.get(), neuronNum);
                        vertex.getValue().setClassFlag(classFlag);
                        Logger.d("Loaded classFlag: " + vertex.getValue().getClassFlag());
                        vertex.voteToHalt();
                        break;
                }
                break;

            case NNMasterCompute.FORWARD_PROPAGATION_STAGE:
                //update activation
                updateActivation(vertex, messages, layerNum, neuronNum);

                //forward propagate
                switch (layerNum) {
                    case OUTPUT_LAYER:
                        //switch to next stage
                        if(neuronNum == 1)
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(1));
                        break;

                    default:
                        //update weights
                        updateFrontWeights(vertex, layerNum);

                        //forward propagate
                        forwardProp(vertex, layerNum);
                        activateBiasUnitOfNextLayer(layerNum);
                        vertex.voteToHalt();
                }

                break;

            case NNMasterCompute.BACKWARD_PROPAGATION_STAGE:
                incrementDataSetIndex(neuronNum, layerNum, dataSetIndex.get());

                switch (layerNum) {
                    case INPUT_LAYER:
                        //calculate derivatives
                        calculateDerivatives(vertex, messages, layerNum);

                        //switch stage to data load
                        if(neuronNum == 1) {
                            aggregate(NNMasterCompute.STAGE_AGG_ID, new IntWritable(-stage.get()
                                    + NNMasterCompute.DATA_LOAD_STAGE));

                            //activate output layer for data loading
                            activateLayer(OUTPUT_LAYER, false);
                        }
                        break;

                    default:
                        calculateError(vertex, messages, layerNum);
                        backPropError(vertex, layerNum, neuronNum);
                        calculateDerivatives(vertex, messages, layerNum);
                        vertex.voteToHalt();
                        break;
                }
                break;

            default:
                vertex.voteToHalt();
        }
    }

    private void generateNetwork() throws IOException {
        Logger.d("Generating network");
        for(int l = INPUT_LAYER; l <= MAX_HIDDEN_LAYER_NUM; l++) {
            int neuronCount = getNeuronCount(l);
            int i = l == INPUT_LAYER ? 1 : 0;
            for (; i <= neuronCount; i++) {
                Text destVertexID = new Text(getNeuronId(l, i));
                NeuronValue nVal = new NeuronValue(i == 0 ? 1d : 0d, 0d, 0d, 0, getNeuronCount(getNextLayerNum(l)));
                addVertexRequest(destVertexID, nVal);
                Logger.d("Generating vertex: " + destVertexID);
            }
        }

        //output layer
        int neuronCount = getNeuronCount(OUTPUT_LAYER);
        for (int i = 1; i <= neuronCount; i++) {
            Text destVertexID = new Text(getNeuronId(OUTPUT_LAYER, i));
            NeuronValue nVal = new NeuronValue(0d, 0d, 0d, 0, 0);
            addVertexRequest(destVertexID, nVal);
            Logger.d("Generating vertex: " + destVertexID);
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
            Edge<Text, DoubleWritable> e = EdgeFactory.create(dstId, new DoubleWritable(weight));
            addEdgeRequest(vertex.getId(), e);
            Logger.d(String.format("Generated edge from %s to %s with weight %f", vertex.getId().toString(),
                    e.getTargetVertexId().toString(), e.getValue().get()));
        }
    }

    private void generateBackEdges(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum) throws IOException {
        Logger.d("Num Edges = " + vertex.getNumEdges());
        for (Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            if (!isAnEdgeToNextLayer(e, layerNum))
                continue;

            Text srcId = new Text(e.getTargetVertexId());
            DoubleWritable weight = new DoubleWritable(e.getValue().get());

            Edge<Text, DoubleWritable> backEdge = EdgeFactory.create(vertex.getId(), weight);
            addEdgeRequest(srcId, backEdge);
            Logger.d(String.format("Generated back edge from %s to %s with weight %f", srcId.toString(),
                    backEdge.getTargetVertexId().toString(), backEdge.getValue().get()));
        }
    }

    private void forwardProp(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum) {
        for (Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            if (!isAnEdgeToNextLayer(e, layerNum))
                continue;

            Text dstId = e.getTargetVertexId();
            Double weight = e.getValue().get();
            Double activation = vertex.getValue().getActivation();
            Double fragment = weight * activation;
            sendMessage(dstId, new Text(fragment + ""));
            Logger.d(String.format("weight: %f, activation: %f", weight, activation));
            Logger.d(String.format("Sending msg to " + dstId + ", msg = " + fragment));
        }
    }

    private void updateActivation(Vertex<Text, NeuronValue, DoubleWritable> vertex, Iterable<Text> messages,
                                  int layerNum, int neuronNum) {

        if(layerNum == INPUT_LAYER || neuronNum == BIAS_UNIT)
            return;

        double weightedInput = 0;

        for(Text m : messages) {
            Logger.d("Incoming msg: " + m);
            Double input = Double.parseDouble(m.toString());
            weightedInput += input;
        }

        double activation = activationFunction(weightedInput);
        vertex.getValue().setWeightedInput(weightedInput);
        vertex.getValue().setActivation(activation);
        Logger.d("New weighted input: " + weightedInput);
        Logger.d("New activation: " + activation);
    }

    private double activationFunction(Double x) {
        Sigmoid sig = new Sigmoid();
        return sig.value(x);
    }

    private double activationFunctionDerivative(Double x) {
        return x*(1-x);
    }

    private void calculateError(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                Iterable<Text> messages, int layerNum) {

        NeuronValue val = vertex.getValue();
        double error;

        switch (layerNum) {
            case OUTPUT_LAYER:
                //calculate using class flag
                error = val.getActivation() - val.getClassFlag();
                val.setError(error);
                break;

            default:
                //calculate using incoming messages
                double weightedError = 0;
                for(Text m : messages) {
                    Logger.d("incoming error fragment: " + m.toString());
                    String[] tokens = m.toString().split(DELIMITER);
                    double fragment = Double.parseDouble(tokens[2]);
                    weightedError += fragment;
                }
                error = weightedError * activationFunctionDerivative(val.getActivation());
                val.setError(error);
        }

        Logger.d("New error: " + val.getError());
    }

    private void calculateDerivatives(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                      Iterable<Text> messages, int layerNum) {

        NeuronValue val = vertex.getValue();
        for(Text m : messages) {
            String[] tokens = m.toString().split(DELIMITER);
            int senderNeuronNum = Integer.parseInt(tokens[0]);
            double senderError = Double.parseDouble(tokens[1]);
            double activation = val.getActivation();
            double derivative = activation*senderError;

            Logger.d(String.format("activation: %.10f, senderError: %.10f", activation, senderError));
            Logger.d(String.format("Setting derivatives[%d] = %.15f", senderNeuronNum-1, derivative));
            val.updateDerivative(senderNeuronNum - 1, derivative);
        }
    }

    private void backPropError(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum, int neuronNum) {
        for(Edge<Text, DoubleWritable> e : vertex.getEdges()) {
            if(isAnEdgeToNextLayer(e, layerNum))
                continue;

            double weight = e.getValue().get();
            Double error = vertex.getValue().getError();
            Double fragment = weight*error;
            String msg = String.format("%s%s%s%s%s", neuronNum, DELIMITER,
                    error.toString(), DELIMITER, fragment.toString());
            sendMessage(e.getTargetVertexId(), new Text(msg));
            Logger.d(String.format("Weight: %f, Error: %f, fragment: %.15f", weight, error, weight*error));
            Logger.d(String.format("Sending message from %s to %s: %s",
                    vertex.getId(), e.getTargetVertexId(), msg));
        }
    }

    private void updateFrontWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int layerNum) {
        IntWritable dataSetIndex = getAggregatedValue(NNMasterCompute.DATA_SET_INDEX_AGG);
        Logger.d("UpdateFrontWeights, dataSetIndex = " + dataSetIndex.get());
        if(dataSetIndex.get() != 1)
            return;

        for(Edge<Text, DoubleWritable> e : vertex.getMutableEdges()) {
            if(!isAnEdgeToNextLayer(e, layerNum))
                continue;

            String[] tokens = e.getTargetVertexId().toString().split(DELIMITER);
            int targetNeuronNum = Integer.parseInt(tokens[1]);
            double derivative = vertex.getValue().getDerivative(targetNeuronNum - 1) / DATA_SIZE;
            updateWeight(vertex, e, derivative, LEARNING_RATE);
        }
    }

    private void updateWeight(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                              Edge<Text, DoubleWritable> e, double derivative, double learningRate) {

        double weight = e.getValue().get();
        double update = learningRate * derivative;
        e.getValue().set(weight - update);
        Logger.d(String.format("Updated weight %s to %s, from %.15f to %.15f. Derivative = %.15f",
                vertex.getId().toString(), e.getTargetVertexId().toString(),
                weight, e.getValue().get(), derivative));
    }

    private void incrementDataSetIndex(int neuronNum, int layerNum, int currentIndex) {
        if(neuronNum == 1 && layerNum == OUTPUT_LAYER) {
            if(currentIndex == DATA_SIZE){
                // set to 1
                aggregate(NNMasterCompute.DATA_SET_INDEX_AGG, new IntWritable(-DATA_SIZE));
            }

            aggregate(NNMasterCompute.DATA_SET_INDEX_AGG, new IntWritable(1));
        }
    }

    public static int getNeuronCount(int layerNum) {
        if (layerNum == OUTPUT_LAYER)
            return LAYER_TO_NEURON[LAYER_TO_NEURON.length - 1];
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

    public static int getNextLayerNum(int layerNum) {
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

    private void activateBiasUnitOfNextLayer(int currentLayerNum) {
        if(currentLayerNum == MAX_HIDDEN_LAYER_NUM)
            return;

        int targetLayerNum = getNextLayerNum(currentLayerNum);
        Text dstId = new Text(getNeuronId(targetLayerNum, BIAS_UNIT));
        sendMessage(dstId, new Text(""));
    }

    private boolean containsBiasUnit(int layerNum) {
        return layerNum != OUTPUT_LAYER;
    }
}

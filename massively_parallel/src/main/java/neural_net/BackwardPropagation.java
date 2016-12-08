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

    PerformanceContext context = null;

    public static void main(String[] args) throws Exception {
        System.exit(ToolRunner.run(new GiraphRunner(), args));
    }

    @Override
    public void compute(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                        Iterable<Text> messages) throws IOException {

        if(context == null)
            context = getWorkerContext();

        if(!context.startTimeRegistered) {
            context.startTime = System.currentTimeMillis();
            context.startTimeRegistered = true;
            Logger.p(String.format("Registered start time: %d", context.startTime));
        }

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

        IntWritable iteration = getAggregatedValue(NumberOfClasses.ITERATIONS_ID);
        IntWritable state = getAggregatedValue(NumberOfClasses.STATE_ID);
        Logger.d("\n\n" + "SS: " + getSuperstep() + "  Vertex ID: " + vertex.getId() + ", Stage: " + state.get());

        if (iteration.get() == Config.MAX_ITER - 1) {
            if (networkNum == 1 && layerNum != Config.OUTPUT_LAYER) {
                finishComputation(vertex, layerNum, neuronNum);
            }
        }

        if(Logger.PERFORMANCE) {
            if (context.prevStage != state.get()) {
                long currTime = System.currentTimeMillis();
                Logger.p(String.format("PrevStage: %d, currStage: %d", context.prevStage, state.get()));
                Logger.p(String.format("StartTime = %d, CurrTime = %d, diff = %.3f secs",
                        context.startTime, currTime, (currTime - context.startTime) / (double) 1000));
                context.prevStage = state.get();
                context.startTime = currTime;
            }
        }

        switch (state.get()) {
            case NumberOfClasses.HIDDEN_LAYER_GENERATION_STATE:
                switch (layerNum) {
                    case Config.MAX_HIDDEN_LAYER_NUM:
                        // set weights array
                        vertex.getValue().generateWeightsArray(getNextLayerNeuronCount(layerNum));

                        setInitWeights(vertex, layerNum, neuronNum);

                        // to next stage
                        if (neuronNum == 1) {
                            turnInputLayerToActive(networkNum);
                            aggregate(NumberOfClasses.NUMBER_OF_NETWORKS_ID, new IntWritable(networkNum));

                            if (networkNum == 1)
                                aggregate(NumberOfClasses.STATE_ID, new IntWritable(1));
                        }
                        break;

                    default:
                        // generate next layer neurons
                        if(neuronNum == 1) {
                            generateBiasUnit(networkNum, getNextLayerNum(layerNum));
                            generateVertices(networkNum, getNextLayerNum(layerNum));
                        }

                        // set weights array
                        vertex.getValue().generateWeightsArray(getNextLayerNeuronCount(layerNum));

                        // set weights
                        setInitWeights(vertex, layerNum, neuronNum);

                }

                vertex.voteToHalt();
                break;

            case NumberOfClasses.FORWARD_PROPAGATION_STATE:
                updateWeights(vertex, networkNum, layerNum, neuronNum);
                updateActivation(vertex, messages, layerNum, neuronNum);

                switch (layerNum) {
                    case Config.OUTPUT_LAYER:
                        aggregateCost(vertex, networkNum, layerNum, neuronNum);

                        // switch state
                        if (networkNum == 1 && neuronNum == 1) {
                            aggregate(NumberOfClasses.STATE_ID, new IntWritable(1));
                        }
                        break;

                    default:
                        activateNextLayerBiasUnit(networkNum, layerNum, neuronNum);
                        forwardProp(vertex, networkNum, layerNum);
                        vertex.voteToHalt();
                }

                break;

            case NumberOfClasses.BACKWARD_PROPAGATION_STATE:
                calculateError(vertex, messages, networkNum, layerNum, neuronNum);

                switch (layerNum) {
                    case Config.INPUT_LAYER:
                        if (networkNum == 1 && neuronNum == 1) {
                            aggregate(NumberOfClasses.STATE_ID, new IntWritable(-state.get()));
                            aggregate(NumberOfClasses.STATE_ID, new IntWritable(NumberOfClasses.FORWARD_PROPAGATION_STATE));
                            aggregate(NumberOfClasses.ITERATIONS_ID, new IntWritable(1));
                        }
                        break;

                    default:
                        backPropagateError(vertex, networkNum, layerNum, neuronNum);
                        vertex.voteToHalt();
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
            Logger.d("ACTIVATION ZERO");
            return 0;
        }
        double fragment = y * Math.log(activation) + (1 - y) * Math.log(1 - activation);

        if (fragment > 0) {
            System.out.printf("positive cost found for vertex: %s \nactivation = %f, cost = %f, y = %d\n",
                    vertex.getId(), activation, fragment, y);
        }

        Logger.d(String.format("y: %d, activation: %f", y, activation));
        Logger.d(String.format("fragment = %f", fragment));
        return fragment;
    }

    private void updateWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex, int networkNum,
                               int layerNum, int neuronNum) {

        if (layerNum == Config.OUTPUT_LAYER)
            return;

        IntWritable m = getAggregatedValue(NumberOfClasses.NUMBER_OF_NETWORKS_ID);
        String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
        Logger.d("Getting error aggregator: " + aggName);
        DoubleDenseVector gradients = getAggregatedValue(aggName);

        Logger.d("DELTA: ");
        if(Logger.DEBUG) {
            for (int i = 0; i < getNextLayerNeuronCount(layerNum); i++) {
                Logger.d(gradients.get(i) + "  ");
            }
        }

        for(int i = 0; i < getNextLayerNeuronCount(layerNum); i++) {
            double gradient = gradients.get(i) / m.get();
            double old = vertex.getValue().getWeight(i);
            double update = gradient * Config.LEARNING_RATE;

            // gradient descent
            vertex.getValue().setWeight(old - update, i);

            Logger.d("Updating front edge " + vertex.getId() + " --> " + getNeuronId(networkNum, getNextLayerNum(layerNum), i+1));
            Logger.d(String.format("Old val: %f, gradient: %f, update: %f, new val: %f",
                    old, gradient, update, vertex.getValue().getWeight(i)));
        }
    }

    private void backPropagateError(Vertex<Text, NeuronValue, DoubleWritable> vertex, int networkNum,
                                    int layerNum, int neuronNum) {

        if(neuronNum == Config.BIAS_UNIT)
            return;

        for(int i = 0; i <= getPrevLayerNeuronCount(layerNum); i++) {
            Text dstId = getNeuronId(networkNum, getPrevLayerNum(layerNum), i);
            double error = vertex.getValue().getError();
            String msg = String.format("%d%s%s", neuronNum, Config.DELIMITER, error);
            sendMessage(dstId, new Text(msg));
            Logger.d(String.format("error: %f", error));
            Logger.d("Sending msg to " + dstId + ", msg = " + msg);
        }
    }

    private void forwardProp(Vertex<Text, NeuronValue, DoubleWritable> vertex, int networkNum,
                             int layerNum) {

        if (layerNum == Config.OUTPUT_LAYER)
            return;

        for(int i = 1; i <= getNextLayerNeuronCount(layerNum); i++) {
            Text dstId = getNeuronId(networkNum, getNextLayerNum(layerNum), i);

            double weight = vertex.getValue().getWeight(i - 1);
            double activation = vertex.getValue().getActivation();
            double fragment = weight * activation;

            String msg = fragment + "";
            sendMessage(dstId, new Text(msg));
            Logger.d(String.format("weight: %f, activation: %f", weight, activation));
            Logger.d("Sending msg to " + dstId + ", msg = " + msg);
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

    private void generateBiasUnit(int networkNum, int layerNum) throws IOException {
        Text id = getNeuronId(networkNum, layerNum, Config.BIAS_UNIT);
        Logger.d("Generating bias unit " + id);
        NeuronValue val = new NeuronValue(1d, 0d, 0d, 0);
        addVertexRequest(id, val);
    }

    private void generateVertices(int networkNum, int layerNum) throws IOException {
        int neuronCount = getNeuronCountByLayer(layerNum);

        for(int i = 1; i <= neuronCount; i++) {
            Text dstId = getNeuronId(networkNum, layerNum, i);
            sendMessage(dstId, new Text(""));
            Logger.d("Generating vertex: " + dstId);
        }
    }

    private void setInitWeights(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                int layerNum, int neuronNum) {

        DoubleDenseVector weights = getAggregatedValue(
                NumberOfClasses.GetWeightAggregatorName(layerNum, neuronNum));

        // set weights
        for(int i = 0; i < getNextLayerNeuronCount(layerNum); i++) {
            double weight = weights.get(i);
            vertex.getValue().setWeight(weight, i);
            Logger.d("Setting init weight to neuron " + (i+1) + ", value = " + weight);
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
        return sig.value(x) * (1 - sig.value(x));
    }

    private void turnInputLayerToActive(int networkNum) {
        for (int i = 0; i <= Config.INPUT_LAYER_NEURON_COUNT; i++) {
            Text dstId = new Text(String.format("%d:%d:%d", networkNum, Config.INPUT_LAYER, i));
            sendMessage(dstId, new Text(""));
        }
    }

    private void finishComputation(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                   int layerNum, int neuronNum) {

        Logger.d("Finishing computation");

        //aggregate the weights
        Logger.d("Aggregating weights for vertex " + vertex.getId());
        String aggName = NumberOfClasses.GetWeightAggregatorName(layerNum, neuronNum);
        DoubleDenseVector weights = getAggregatedValue(aggName);

        //flush the aggregator
        int vecSize = getNextLayerNeuronCount(layerNum);
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

    public static int getNextLayerNum(int layerNum) {
        switch (layerNum) {
            case Config.OUTPUT_LAYER:
                return Config.INPUT_LAYER;
            case Config.MAX_HIDDEN_LAYER_NUM:
                return Config.OUTPUT_LAYER;
            default:
                return layerNum + 1;
        }
    }

    public static int getPrevLayerNum(int layerNum) {
        switch (layerNum) {
            case Config.INPUT_LAYER:
                return Config.OUTPUT_LAYER;
            case Config.OUTPUT_LAYER:
                return Config.MAX_HIDDEN_LAYER_NUM;
            default:
                return layerNum - 1;
        }
    }

    public Text getNeuronId(int networkNum, int layerNum, int neuronNum) {
        String id = String.format("%d%s%d%s%d", networkNum, Config.DELIMITER, layerNum,
                Config.DELIMITER, neuronNum);
        return new Text(id);
    }

    private void generateNextLayerBiasUnit(int networkNum, int layerNum, int neuronNum) throws IOException {
        // run only once per network
        if (neuronNum == Config.BIAS_UNIT && layerNum != Config.MAX_HIDDEN_LAYER_NUM) {
            Text id = getNeuronId(networkNum, getNextLayerNum(layerNum), Config.BIAS_UNIT);
            Logger.d("Generating bias unit " + id);
            NeuronValue val = new NeuronValue(1d, 0d, 0d, 0);
            addVertexRequest(id, val);
        }
    }

    public static int getNeuronCountByLayer(int layerNum) {
        switch (layerNum) {
            case Config.OUTPUT_LAYER:
                return Config.LAYER_COUNTS[Config.LAYER_COUNTS.length - 1];
            default:
                return Config.LAYER_COUNTS[layerNum - 1];
        }
    }

    public static int getPrevLayerNeuronCount(int layerNum) {
        return getNeuronCountByLayer(getPrevLayerNum(layerNum));
    }

    public static int getNextLayerNeuronCount(int layerNum) {
        return getNeuronCountByLayer(getNextLayerNum(layerNum));
    }

    private void activateNextLayerBiasUnit(int networkNum, int layerNum, int neuronNum) {
        if (neuronNum == 1 && layerNum != Config.MAX_HIDDEN_LAYER_NUM) {
            Text id = getNeuronId(networkNum, getNextLayerNum(layerNum), Config.BIAS_UNIT);
            sendMessage(id, new Text(""));
            Logger.d("Activating unit: " + id);
        }
    }

    private void aggregateCost(Vertex<Text, NeuronValue, DoubleWritable> vertex, int networkNum, int layerNum,
                               int neuronNum) {

        if (layerNum != Config.OUTPUT_LAYER)
            return;

        if (networkNum == 1 && neuronNum == 1) {
            // flush aggregator (only once)
            DoubleWritable oldCost = getAggregatedValue(NumberOfClasses.COST_AGGREGATOR);
            aggregate(NumberOfClasses.COST_AGGREGATOR, new DoubleWritable(-oldCost.get()));
        }

        double cost = neuronCost(vertex);
        Logger.d("Cost frag = " + cost);
        aggregate(NumberOfClasses.COST_AGGREGATOR, new DoubleWritable(cost));
    }

    private void updateActivation(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                  Iterable<Text> messages, int layerNum, int neuronNum) {

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
    }

    private void calculateError(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                                Iterable<Text> messages, int networkNum, int layerNum, int neuronNum) {

        Double error;

        switch (layerNum) {
            case Config.OUTPUT_LAYER:
                error = vertex.getValue().getActivation() - vertex.getValue().getClassFlag();
                vertex.getValue().setError(error);
                Logger.d("New error: " + vertex.getValue().getError());
                break;

            default:
                //flush error aggregator DELTA
                if (networkNum == 1) {
                    Logger.d(String.format("flushing error agg for layerNum: %s, neuronNum: %d", layerNum, neuronNum));

                    int nextLayerNeuronCount = getNeuronCountByLayer(getNextLayerNum(layerNum));
                    if (layerNum == Config.MAX_HIDDEN_LAYER_NUM)
                        nextLayerNeuronCount = Config.OUTPUT_LAYER_NEURON_COUNT;

                    // reset error accumulator to zero
                    flushErrorAggregator(layerNum, neuronNum, nextLayerNeuronCount);
                }

                double weightedError = 0d;
                DoubleDenseVector errVector = new DoubleDenseVector(getNextLayerNeuronCount(layerNum));

                for (Text m : messages) {
                    Logger.d("message   : " + m);
                    String[] msgTokens = m.toString().split(Config.DELIMITER);
                    int senderNeuronNum = Integer.parseInt(msgTokens[0]);
                    double activation = vertex.getValue().getActivation();
                    double srcError = Double.parseDouble(msgTokens[1]);
                    double weight = vertex.getValue().getWeight(senderNeuronNum - 1);
                    double input = srcError * weight;

                    errVector.set(senderNeuronNum - 1, activation * srcError);      // to calculate gradient later
                    weightedError += input;
                    Logger.d(String.format("senderNeuronNum: %d, srcError: %s, weight: %s, input: %s",
                            senderNeuronNum, Double.toString(srcError), Double.toString(weight), Double.toString(input)));
                }

                String aggName = NumberOfClasses.GetErrorAggregatorName(layerNum, neuronNum);
                aggregate(aggName, errVector);

                int size = getNextLayerNeuronCount(layerNum);
                Logger.d("errVector : ");
                if(Logger.DEBUG) {
                    for (int i = 0; i < size; i++) {
                        Logger.d(errVector.get(i) + "  ");
                    }
                }

                // calculate error
                if (layerNum != Config.INPUT_LAYER) {
                    Double weightedInput = vertex.getValue().getWeightedInput();
                    error = weightedError * activationFunctionDerivative(weightedInput);
                    vertex.getValue().setError(error);
                    Logger.d("Error     : " + error);
                }
        }
    }

    private void activateNextLayer(int networkNum, int layerNum) {
        int i = layerNum == Config.MAX_HIDDEN_LAYER_NUM ? 1 : 0;
        int nextLayer = getNextLayerNum(layerNum);
        for (; i <= getNextLayerNeuronCount(layerNum); i++) {
            Text dstId = new Text(getNeuronId(networkNum, nextLayer, i));
            Logger.d("Activating " + dstId);
            sendMessage(dstId, new Text(""));
        }
    }

    private int getNeuronNumFromId(Text id) {
        String[] tokens = id.toString().split(Config.DELIMITER);
        return Integer.parseInt(tokens[2]);
    }

    private int getLayerNumFromId(Text id) {
        String[] tokens = id.toString().split(Config.DELIMITER);
        return Integer.parseInt(tokens[1]);
    }

    private void updateWeight(Vertex<Text, NeuronValue, DoubleWritable> vertex,
                              Text targetVertexId, double derivative, double learningRate) {

        double oldWeight = vertex.getEdgeValue(new Text(targetVertexId)).get();
        double update = learningRate * derivative;
        vertex.setEdgeValue(targetVertexId, new DoubleWritable(oldWeight - update));
        Logger.d(String.format("Updated weight %s to %s, from %s to %s. Derivative = %s",
                vertex.getId().toString(), targetVertexId.toString(),
                Double.toString(oldWeight),
                Double.toString(vertex.getEdgeValue(new Text(targetVertexId)).get()),
                Double.toString(derivative)));
    }

}

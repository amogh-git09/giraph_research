package neural_net;

import org.apache.giraph.aggregators.IntMaxAggregator;
import org.apache.giraph.aggregators.IntOverwriteAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVector;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVectorSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.hadoop.io.IntWritable;

import java.util.Random;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class NumberOfClasses extends DefaultMasterCompute {

    public static final int HIDDEN_LAYER_GENERATION_STATE = 0;
    public static final int FORWARD_PROPAGATION_STATE = 1;

    public static final String WEIGHT_AGGREGATOR_PREFIX = "weightAggregator";
    public static final String STATE_ID = "StateAggregator";
    public static final String NUMBER_OF_NETWORKS_ID = "NumberOfNetworksAggregator";
    public static final double EPSILON = 0.2;
    public static final Random random = new Random();

    @Override
    public void compute() {
        if(getSuperstep() > 10) {
            haltComputation();
        }

        System.out.print("\nSS: " + getSuperstep());
        IntWritable state = getAggregatedValue(STATE_ID);
        switch (state.get()) {
            case HIDDEN_LAYER_GENERATION_STATE: System.out.println("  HIDDEN LAYER GENERATION STAGE");
                break;
            case FORWARD_PROPAGATION_STATE: System.out.println("  FORWARD PROPAGATION STAGE");
                break;
            default: System.out.println("  UNKNOWN STAGE " + state.get());
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STATE_ID, IntOverwriteAggregator.class);
        registerPersistentAggregator(NUMBER_OF_NETWORKS_ID, IntMaxAggregator.class);
        registerAndInitializeWeightAggregators(BackwardPropagation.MAX_HIDDEN_LAYER_NUM,
                BackwardPropagation.INPUT_LAYER_NEURON_COUNT, BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT);

        initializeLayerWeightAggs(BackwardPropagation.MAX_HIDDEN_LAYER_NUM,
                BackwardPropagation.INPUT_LAYER_NEURON_COUNT,
                BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT,
                BackwardPropagation.OUTPUT_LAYER_NEURON_COUNT);
        setAggregatedValue(STATE_ID, new IntWritable(HIDDEN_LAYER_GENERATION_STATE));
    }

    private void initializeLayerWeightAggs(int finalHiddenLayerNum,
                                           int inputLayerNeuronCount,
                                           int hiddenLayerNeuronCount,
                                           int outputLayerNeuronCount) {

        //input layer
        initializeLayerWeightAgg(NeuralNetworkVertexInputFormat.INPUT_LAYER,
                inputLayerNeuronCount, hiddenLayerNeuronCount);

        //hidden layers
        for(int i=2; i<finalHiddenLayerNum; i++) {
            initializeLayerWeightAgg(i, hiddenLayerNeuronCount, hiddenLayerNeuronCount);
        }

        //output layer
        initializeLayerWeightAgg(finalHiddenLayerNum, hiddenLayerNeuronCount, outputLayerNeuronCount);
    }

    private void initializeLayerWeightAgg(int layerNum, int neuronCount, int numOfOutgoingEdges) {
        for(int i=1; i<=neuronCount; i++) {
            DoubleDenseVector vector = new DoubleDenseVector(numOfOutgoingEdges);

            for(int j=0; j<numOfOutgoingEdges; j++) {
                Double initVal = getRandomInRange(-EPSILON, EPSILON);
                vector.set(j, initVal);
            }

            String aggName = GetWeightAggregatorName(layerNum, i);
            setAggregatedValue(aggName, vector);
        }
    }

    public static String GetWeightAggregatorName(int layerNum, int neuronNum) {
        return String.format("%s:%d:%d", WEIGHT_AGGREGATOR_PREFIX, layerNum, neuronNum);
    }

    private void registerAndInitializeWeightAggregators(int finalHiddenLayerNum,
                                                        int inputLayerNeuronCount,
                                                        int hiddenLayerNeuronCount)
            throws InstantiationException, IllegalAccessException {

        registerLayerWeightAgg(NeuralNetworkVertexInputFormat.INPUT_LAYER, inputLayerNeuronCount);

        for(int i=2; i<=finalHiddenLayerNum; i++) {
            registerLayerWeightAgg(i, hiddenLayerNeuronCount);
        }
    }

    private void registerLayerWeightAgg(int layerNum, int neuronCount)
            throws IllegalAccessException, InstantiationException {

        for(int i=1; i<=neuronCount; i++) {
            String aggName = GetWeightAggregatorName(layerNum, i);
            registerPersistentAggregator(aggName, DoubleDenseVectorSumAggregator.class);
        }
    }

    private double getRandomInRange(Double min, Double max) {
        Double rand = random.nextDouble();
        return min + (max - min) * rand;
    }
}

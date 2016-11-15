package neural_net;

import org.apache.giraph.aggregators.DoubleSumAggregator;
import org.apache.giraph.aggregators.IntMaxAggregator;
import org.apache.giraph.aggregators.IntOverwriteAggregator;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVector;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVectorSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

import java.util.Random;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class NumberOfClasses extends DefaultMasterCompute {

    public static final int HIDDEN_LAYER_GENERATION_STATE = 0;
    public static final int BACK_EDGES_GENERATION_STATE = 1;
    public static final int FORWARD_PROPAGATION_STATE = 2;
    public static final int BACKWARD_PROPAGATION_STATE = 3;

    public static final String COST_AGGREGATOR = "costAggregator";
    public static final String WEIGHT_AGGREGATOR_PREFIX = "weightAggregator";
    public static final String ERROR_AGGREGATOR_PREFIX = "errorAggregator";
    public static final String STATE_ID = "StateAggregator";
    public static final String NUMBER_OF_NETWORKS_ID = "NumberOfNetworksAggregator";
    public static final double EPSILON = 0.5;
    public static final Random random = new Random();

    int printCounter = 3;

    @Override
    public void compute() {
        if(getSuperstep() > BackwardPropagation.MAX_ITER) {
            //print weights

            System.out.println("\n-------- HALT PROCESSING -----------\n\n");

            System.out.println("Weights for input layer\n");
            for(int j=0; j<=BackwardPropagation.INPUT_LAYER_NEURON_COUNT; j++) {
                String aggName = GetWeightAggregatorName(1, j);
                DoubleDenseVector weights = getAggregatedValue(aggName);

                for(int k=1; k<=BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT; k++) {
                    Double weight = weights.get(k-1);
                    System.out.printf("%.7f ", weight);
                }
                System.out.println("");
            }
            System.out.println("");

            for(int i=2; i<=BackwardPropagation.MAX_HIDDEN_LAYER_NUM; i++) {
                System.out.printf("Weights for layer %d\n\n", i);

                for(int j=0; j<=BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT; j++) {
                    String aggName = GetWeightAggregatorName(i, j);
                    DoubleDenseVector weights = getAggregatedValue(aggName);

                    int weightSize = i == BackwardPropagation.MAX_HIDDEN_LAYER_NUM ?
                            BackwardPropagation.OUTPUT_LAYER_NEURON_COUNT :
                            BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT;

                    for(int k=1; k<=weightSize; k++) {
                        Double weight = weights.get(k-1);
                        System.out.print(weight + " ");
                    }
                    System.out.println("");
                }
                System.out.println("\n");
            }

            haltComputation();
        }

//        System.out.print("\nSS: " + getSuperstep());
        IntWritable state = getAggregatedValue(STATE_ID);
        switch (state.get()) {
            case HIDDEN_LAYER_GENERATION_STATE: System.out.println("  HIDDEN LAYER GENERATION STAGE");
                break;
            case BACK_EDGES_GENERATION_STATE: System.out.println("  BACK EDGES GENERATION STAGE");
                break;
            case FORWARD_PROPAGATION_STATE:
//                System.out.println("SS: " + getSuperstep() + "  FORWARD PROPAGATION STAGE");
                if(printCounter++ % 3 == 0) {
                    DoubleWritable costWr = getAggregatedValue(COST_AGGREGATOR);
                    IntWritable m = getAggregatedValue(NUMBER_OF_NETWORKS_ID);
                    double cost = -costWr.get() / m.get();
                    System.out.println("SS: " + getSuperstep() + ", Cost at master = " + cost);
                    if(cost < 0) {
                        haltComputation();
                    }
                }
                break;
            case BACKWARD_PROPAGATION_STATE:
//                System.out.println("  BACKWARD PROPAGATION STAGE");

//                for(int l = 1; l<=BackwardPropagation.MAX_HIDDEN_LAYER_NUM; l++) {
//                    String aggName = GetErrorAggregatorName(l, 1);
//                    System.out.println("The error vector for '" + aggName + "' is ");
//                    DoubleDenseVector vec = getAggregatedValue(aggName);
//                    for(int i=0; i<BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT; i++) {
//                        System.out.print(vec.get(i) + "  ");
//                    }
//                    System.out.println("");
//                }

                break;
            default: System.out.println("  UNKNOWN STAGE " + state.get());
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        System.out.println("Registering all the aggregators");
        registerPersistentAggregator(STATE_ID, IntOverwriteAggregator.class);
        registerPersistentAggregator(NUMBER_OF_NETWORKS_ID, IntMaxAggregator.class);
        registerPersistentAggregator(COST_AGGREGATOR, DoubleSumAggregator.class);
        registerWeightAggregators(BackwardPropagation.MAX_HIDDEN_LAYER_NUM,
                BackwardPropagation.INPUT_LAYER_NEURON_COUNT, BackwardPropagation.HIDDEN_LAYER_NEURON_COUNT);
        registerErrorAggregators(BackwardPropagation.MAX_HIDDEN_LAYER_NUM,
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
        for(int i=0; i<=neuronCount; i++) {
            DoubleDenseVector vector = new DoubleDenseVector(numOfOutgoingEdges);

            for(int j=0; j<numOfOutgoingEdges; j++) {
                Double initVal = getRandomInRange(-EPSILON, EPSILON);

//                switch (layerNum) {
//                    case 1:
//                        switch (i) {
//                            case 0:
//                                switch (j) {
//                                    case 0: initVal = -0.051; break;
//                                    case 1: initVal = 0.002; break;
//                                }
//                                break;
//                            case 1:
//                                switch (j) {
//                                    case 0: initVal = 0.003; break;
//                                    case 1: initVal = 0.016; break;
//                                }
//                                break;
//                            case 2:
//                                switch (j) {
//                                    case 0: initVal = 0.071; break;
//                                    case 1: initVal = 0.049; break;
//                                }
//                                break;
//                        }
//                        break;
//
//                    case 2:
//                        switch (i) {
//                            case 0:
//                                switch (j) {
//                                    case 0: initVal = 0.012; break;
//                                }
//                                break;
//                            case 1:
//                                switch (j) {
//                                    case 0: initVal = -0.163; break;
//                                }
//                                break;
//                            case 2:
//                                switch (j) {
//                                    case 0: initVal = 0.058; break;
//                                }
//                                break;
//                        }
//                        break;
//                }

                vector.set(j, initVal);
            }

            String aggName = GetWeightAggregatorName(layerNum, i);
            setAggregatedValue(aggName, vector);
        }
    }

    public static String GetWeightAggregatorName(int layerNum, int neuronNum) {
        return String.format("%s:%d:%d", WEIGHT_AGGREGATOR_PREFIX, layerNum, neuronNum);
    }

    private void registerWeightAggregators(int finalHiddenLayerNum,
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

        for(int i=0; i<=neuronCount; i++) {
            String aggName = GetWeightAggregatorName(layerNum, i);
            registerPersistentAggregator(aggName, DoubleDenseVectorSumAggregator.class);
        }
    }

    public static String GetErrorAggregatorName(int layerNum, int neuronNum) {
        return String.format("%s:%d:%d", ERROR_AGGREGATOR_PREFIX, layerNum, neuronNum);
    }

    private void registerErrorAggregators(int finalHiddenLayerNum,
                                          int inputLayerNeuronCount,
                                          int hiddenLayerNeuronCount)
            throws InstantiationException, IllegalAccessException {

        registerErrorAgg(NeuralNetworkVertexInputFormat.INPUT_LAYER, inputLayerNeuronCount);

        for(int i=2; i<=finalHiddenLayerNum; i++) {
            registerErrorAgg(i, hiddenLayerNeuronCount);
        }
    }

    private void registerErrorAgg(int layerNum, int neuronCount)
            throws IllegalAccessException, InstantiationException {

        for(int i=0; i<=neuronCount; i++) {
            String aggName = GetErrorAggregatorName(layerNum, i);
            registerPersistentAggregator(aggName, DoubleDenseVectorSumAggregator.class);
        }
    }

    private double getRandomInRange(Double min, Double max) {
        Double rand = random.nextDouble();
        return min + (max - min) * rand;
    }
}

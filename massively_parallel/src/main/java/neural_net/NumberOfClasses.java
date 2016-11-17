package neural_net;

import org.apache.giraph.aggregators.DoubleSumAggregator;
import org.apache.giraph.aggregators.IntMaxAggregator;
import org.apache.giraph.aggregators.IntOverwriteAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
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
    public static final String ITERATIONS_ID = "IterationsAgg";
    public static final double EPSILON = 0.5;
    public static final Random random = new Random();

    int printCounter = 3;
    int prevIteraion = 0;

    @Override
    public void compute() {
        IntWritable iteration = getAggregatedValue(ITERATIONS_ID);
        if (iteration.get() > Config.MAX_ITER) {
            printWeights();
            haltComputation();
            return;
        }

        printCost();

        IntWritable state = getAggregatedValue(STATE_ID);
        switch (state.get()) {
            case HIDDEN_LAYER_GENERATION_STATE:
                Logger.d("HIDDEN LAYER GENERATION STAGE");
                break;
            case BACK_EDGES_GENERATION_STATE:
                Logger.d("BACK EDGES GENERATION STAGE");
                break;
            case FORWARD_PROPAGATION_STATE:
                Logger.d("FORWARD PROPAGATION STAGE");
                break;
            case BACKWARD_PROPAGATION_STATE:
                Logger.d("BACKWARD PROPAGATION STAGE");
                printErrorVector();
                break;
            default:
                Logger.d("  UNKNOWN STAGE " + state.get());
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        Logger.d("Registering all the aggregators");
        registerPersistentAggregator(STATE_ID, IntSumAggregator.class);
        registerPersistentAggregator(NUMBER_OF_NETWORKS_ID, IntMaxAggregator.class);
        registerPersistentAggregator(COST_AGGREGATOR, DoubleSumAggregator.class);
        registerPersistentAggregator(ITERATIONS_ID, IntSumAggregator.class);
        registerWeightAggregators();
        registerErrorAggregators();

        initializeLayerWeightAggs();
        setAggregatedValue(STATE_ID, new IntWritable(HIDDEN_LAYER_GENERATION_STATE));
    }

    private void initializeLayerWeightAggs() {
        for(int i = Config.INPUT_LAYER; i <= Config.MAX_HIDDEN_LAYER_NUM; i++) {
            initializeLayerWeightAgg(i, BackwardPropagation.getNeuronCountByLayer(i),
                    BackwardPropagation.getNextLayerNeuronCount(i));
        }
    }

    private void initializeLayerWeightAgg(int layerNum, int neuronCount, int numOfOutgoingEdges) {
        for (int i = 0; i <= neuronCount; i++) {
            DoubleDenseVector vector = new DoubleDenseVector(numOfOutgoingEdges);

            for (int j = 0; j < numOfOutgoingEdges; j++) {
                if(Config.TEST_WEIGHTS) {
                    setTestWeights(layerNum, i, j, vector);
                } else {
                    Double initVal = getRandomInRange(-EPSILON, EPSILON);
                    vector.set(j, initVal);
                }
            }

            String aggName = GetWeightAggregatorName(layerNum, i);
            setAggregatedValue(aggName, vector);
        }
    }

    public static String GetWeightAggregatorName(int layerNum, int neuronNum) {
        return String.format("%s:%d:%d", WEIGHT_AGGREGATOR_PREFIX, layerNum, neuronNum);
    }

    private void registerWeightAggregators()
            throws InstantiationException, IllegalAccessException {

        for(int i = Config.INPUT_LAYER; i <= Config.MAX_HIDDEN_LAYER_NUM; i++) {
            registerLayerWeightAgg(i, BackwardPropagation.getNeuronCountByLayer(i));
        }
    }

    private void registerLayerWeightAgg(int layerNum, int neuronCount)
            throws IllegalAccessException, InstantiationException {

        for (int i = 0; i <= neuronCount; i++) {
            String aggName = GetWeightAggregatorName(layerNum, i);
            registerPersistentAggregator(aggName, DoubleDenseVectorSumAggregator.class);
        }
    }

    public static String GetErrorAggregatorName(int layerNum, int neuronNum) {
        return String.format("%s:%d:%d", ERROR_AGGREGATOR_PREFIX, layerNum, neuronNum);
    }

    private void registerErrorAggregators()
            throws InstantiationException, IllegalAccessException {

        for(int i = Config.INPUT_LAYER; i <= Config.MAX_HIDDEN_LAYER_NUM; i++) {
            registerErrorAgg(i, BackwardPropagation.getNeuronCountByLayer(i));
        }
    }

    private void registerErrorAgg(int layerNum, int neuronCount)
            throws IllegalAccessException, InstantiationException {

        for (int i = 0; i <= neuronCount; i++) {
            String aggName = GetErrorAggregatorName(layerNum, i);
            Logger.d("Registering error aggregator: " + aggName);
            registerPersistentAggregator(aggName, DoubleDenseVectorSumAggregator.class);
        }
    }

    private double getRandomInRange(Double min, Double max) {
        Double rand = random.nextDouble();
        return min + (max - min) * rand;
    }

    private void printWeights() {
        //print weights
        System.out.println("\n-------- HALT PROCESSING -----------\n\n");
        System.out.println("Weights for input layer\n");

        for (int i = Config.INPUT_LAYER; i <= Config.MAX_HIDDEN_LAYER_NUM; i++) {
            System.out.printf("Weights for layer %d\n-----------\n", i);
            for(int j = 0; j <= BackwardPropagation.getNeuronCountByLayer(i); j++) {
                String aggName = GetWeightAggregatorName(i, j);
                DoubleDenseVector weights = getAggregatedValue(aggName);

                int weightSize = BackwardPropagation.getNextLayerNeuronCount(i);
                for (int k = 1; k <= weightSize; k++) {
                    Double weight = weights.get(k - 1);
                    System.out.print(weight + " ");
                }
                System.out.println("\n");
            }
            System.out.println("\n");
        }
    }

    private void printErrorVector() {
        for (int l = Config.INPUT_LAYER; l <= Config.MAX_HIDDEN_LAYER_NUM; l++) {
            String aggName = GetErrorAggregatorName(l, 1);
            Logger.d("The error vector for '" + aggName + "' is ");
            DoubleDenseVector vec = getAggregatedValue(aggName);
            for (int i = 0; i < BackwardPropagation.getNeuronCountByLayer(l); i++) {
                if (Logger.DEBUG)
                    System.out.print(vec.get(i) + "  ");
            }
            if (Logger.DEBUG)
                System.out.println("");
        }
    }

    private void setTestWeights(int layerNum, int i, int j, DoubleDenseVector vector) {
        double initVal = 0;
        switch (layerNum) {
            case 1:
                switch (i) {
                    case 0:
                        switch (j) {
                            case 0:
                                initVal = -0.051;
                                break;
                            case 1:
                                initVal = 0.002;
                                break;
                        }
                        break;

                    case 1:
                        switch (j) {
                            case 0:
                                initVal = 0.003;
                                break;
                            case 1:
                                initVal = 0.016;
                                break;
                        }
                        break;
                    case 2:
                        switch (j) {
                            case 0:
                                initVal = 0.071;
                                break;
                            case 1:
                                initVal = 0.049;
                                break;
                        }
                        break;
                }
                break;

            case 2:
                switch (i) {
                    case 0:
                        switch (j) {
                            case 0:
                                initVal = 0.012;
                                break;
                        }
                        break;
                    case 1:
                        switch (j) {
                            case 0:
                                initVal = -0.163;
                                break;
                        }
                        break;
                    case 2:
                        switch (j) {
                            case 0:
                                initVal = 0.058;
                                break;
                        }
                        break;
                }
                break;
        }

        vector.set(j, initVal);
    }

    private void printCost() {
        IntWritable iteration = getAggregatedValue(ITERATIONS_ID);

        if(prevIteraion < iteration.get()) {
            prevIteraion = iteration.get();
            DoubleWritable costWr = getAggregatedValue(COST_AGGREGATOR);
            IntWritable m = getAggregatedValue(NUMBER_OF_NETWORKS_ID);
            double cost = -costWr.get() / m.get();
            Logger.i("iteration: " + iteration.get() + ", Cost at master = " + cost);
            if(cost < 0) {
                haltComputation();
            }
        }
    }
}

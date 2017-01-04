import no.uib.cipr.matrix.DenseMatrix;
import org.apache.giraph.aggregators.DoubleSumAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.hadoop.io.IntWritable;

import java.util.Random;

/**
 * Created by amogh09 on 16/12/25.
 */
public class NNMasterCompute extends DefaultMasterCompute {
    public static final int HIDDEN_LAYER_GENERATION_STAGE = 0;
    public static final int FORWARD_PROPAGATION_STAGE = 1;
    public static final int BACKWARD_PROPAGATION_STAGE = 2;
    public static final int WEIGHT_UPDATE_STAGE = 3;

    public static final String STAGE_ID = "StageAggregator";
    public static final String COST_ID = "CostAggregator";
    public static final String DATANUM_ID = "DataAggregator";
    public static final String ITERATION_ID = "IterAggregator";

    @Override
    public void compute() {
        IntWritable iterations = getAggregatedValue(ITERATION_ID);

        if(getSuperstep() == 0) {
            Logger.i("DataSize = " + Config.dataSize);
        }

        if(iterations.get() > Config.MAX_ITER) {
            Logger.i("Superstep: " + getSuperstep());
            IntWritable dataSize = getAggregatedValue(DATANUM_ID);
            Logger.p("DataSize = " + dataSize.get() + " instances");

            Logger.i("Weights:");

            for(int i=Config.INPUT; i != Config.OUTPUT; i = Backpropagation.getNextLayerNum(i)) {
                String aggName = getWeightAggregatorName(i);
                DenseMatrixWritable m = getAggregatedValue(aggName);
                System.out.println("Weights for layer: " + i);
                Backpropagation.printMatrix(m.getMatrix(), true);
                System.out.println("\n");
            }

            haltComputation();
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STAGE_ID, IntSumAggregator.class);
        registerPersistentAggregator(COST_ID, DoubleSumAggregator.class);
        registerPersistentAggregator(DATANUM_ID, IntSumAggregator.class);
        registerPersistentAggregator(ITERATION_ID, IntSumAggregator.class);

        registerWeightMatrices();
        registerDeltaMatrices();

        initializeWeightMatrices();
        initializeDeltaMatrices();
        setAggregatedValue(STAGE_ID, new IntWritable(HIDDEN_LAYER_GENERATION_STAGE));
    }

    private void registerWeightMatrices() throws IllegalAccessException, InstantiationException {
        for(int i = Config.INPUT; i != Config.OUTPUT; i = Backpropagation.getNextLayerNum(i)) {
            String aggName = getWeightAggregatorName(i);
            Logger.d("Registering weight aggregator: " + aggName);
            registerPersistentAggregator(aggName, DenseMatrixWritableSumAggregator.class);
        }
    }

    private void initializeWeightMatrices() {
        for(int i = Config.INPUT; i != Config.OUTPUT; i = Backpropagation.getNextLayerNum(i)) {
            String aggName = getWeightAggregatorName(i);
            int rows = Backpropagation.getNextLayerNeuronCount(i);
            int cols = Backpropagation.getNeuronCount(i);
            DenseMatrix matrix = generateRandomMatrix(Backpropagation.getNextLayerNeuronCount(i),
                    Backpropagation.getNeuronCount(i));
            Logger.d(String.format("Fetching rand matrix of size %dx%d", rows, cols));
            setAggregatedValue(aggName, new DenseMatrixWritable(matrix));
            Backpropagation.printMatrix(matrix);
        }
    }

    private void registerDeltaMatrices() throws IllegalAccessException, InstantiationException {
        for(int i = Config.INPUT; i != Config.OUTPUT; i = Backpropagation.getNextLayerNum(i)) {
            String aggName = getDeltaAggregatorName(i);
            registerPersistentAggregator(aggName, DenseMatrixWritableSumAggregator.class);
        }
    }

    private void initializeDeltaMatrices() {
        for(int i = Config.INPUT; i != Config.OUTPUT; i = Backpropagation.getNextLayerNum(i)) {
            String aggName = getDeltaAggregatorName(i);
            int rows = Backpropagation.getNextLayerNeuronCount(i);
            int cols = Backpropagation.getNeuronCount(i);
            DenseMatrix matrix = new DenseMatrix(rows, cols);
            Logger.d(String.format("Registering %s with size %dx%d", aggName, rows, cols));
            Backpropagation.printMatrix(matrix);
            setAggregatedValue(aggName, new DenseMatrixWritable(matrix));
        }
    }

    public static String getWeightAggregatorName(int layerNum) {
        return String.format("WeightAggregator%s%d", Config.DELIMITER, layerNum);
    }

    public static String getDeltaAggregatorName(int layerNum) {
        return String.format("DeltaAggregator%s%d", Config.DELIMITER, layerNum);
    }

    private static DenseMatrix generateRandomMatrix(int rows, int cols) {
        double min = - Config.EPSILON;
        double max = Config.EPSILON;
        Random r = new Random();
        double[][] data = new double[rows][cols];

        if (Config.TESTING) {
            if(rows == 3) {
                double[][] testData = {{1, 1, 1},
                        {-0.051, 0.003, 0.071},
                        {0.002, 0.016, 0.049}};
                return new DenseMatrix(testData);
            }
            else if(rows == 1) {
                double[][] testData = {{0.012, -0.163, 0.058}};
                return new DenseMatrix(testData);
            }
        }

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                data[i][j] = min + (max - min) * r.nextDouble();
            }
        }

        return new DenseMatrix(data);
    }
}


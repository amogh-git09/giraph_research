import no.uib.cipr.matrix.DenseMatrix;
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

    public static final String STAGE_ID = "StageAggregator";

    @Override
    public void compute() {
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STAGE_ID, IntSumAggregator.class);

        registerWeightMatrices();
        initializeWeightMatrices();
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
            Logger.d("Initializing " + aggName);
            DenseMatrix matrix = generateRandomMatrix(Config.ARCHITECTURE[
                    Backpropagation.getNextLayerNum(i)], Config.ARCHITECTURE[i]);
            setAggregatedValue(aggName, new DenseMatrixWritable(matrix));
            Backpropagation.printMatrix(matrix);
        }
    }

    public static String getWeightAggregatorName(int layerNum) {
        return String.format("WeightAggregator%s%d", Config.DELIMITER, layerNum);
    }

    private static DenseMatrix generateRandomMatrix(int rows, int cols) {
        double min = - Config.EPSILON;
        double max = Config.EPSILON;
        Random r = new Random();
        double[][] data = new double[rows][cols];

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                data[i][j] = min + (max - min) * r.nextDouble();
            }
        }

        return new DenseMatrix(data);
    }
}


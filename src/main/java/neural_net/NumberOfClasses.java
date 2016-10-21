package neural_net;

import org.apache.giraph.aggregators.IntMaxAggregator;
import org.apache.giraph.aggregators.IntOverwriteAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseMatrix;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVectorSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.hadoop.io.IntWritable;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class NumberOfClasses extends DefaultMasterCompute {
    public static final int HIDDEN_LAYER_GENERATION = 0;

    public static final String STATE_ID = "StateAggregator";
    public static final String NUMBER_OF_CLASSES_ID = "NumberOfClassesAggregator";
    public static final String NUMBER_OF_INPUT_NEUTRONS_ID = "NumberOfInputNeutronsAggregator";
    public static final String NUMBER_OF_NETWORKS_ID = "NumberOfNetworksAggregator";
    public static final String NUMBER_OF_HIDDEN_LAYERS_ID = "NumberOfHiddenLayers";
    public static final String HIDDEN_LAYER_NEURON_COUNT = "HiddenLayerNeuronCount";
    public static final double EPSILON = 0.2;

    public static final Random random = new Random();

    public ArrayList<PersistentDoubleDenseMatrixSumAggregator> weightAggregators = new ArrayList<>();

    @Override
    public void compute() {
        if (getSuperstep() == 0) {
            //create required weight aggregators
            IntWritable aggregatorCount = getAggregatedValue(NUMBER_OF_HIDDEN_LAYERS_ID);
            for (int i = 0; i <= aggregatorCount.get(); i++) {
                weightAggregators.add(new PersistentDoubleDenseMatrixSumAggregator("weightMatrix" + i));
            }

            IntWritable hiddenLayerNeuronCount = getAggregatedValue(HIDDEN_LAYER_NEURON_COUNT);
            IntWritable numberOfClasses = getAggregatedValue(NUMBER_OF_CLASSES_ID);
            IntWritable inputLayerNeuronCount = getAggregatedValue(NUMBER_OF_INPUT_NEUTRONS_ID);
            DoubleDenseMatrix matrix;

            //register the weight aggregators
            try {
                for (int i = 0; i < aggregatorCount.get(); i++) {
                    weightAggregators.get(i).registerPersistent(hiddenLayerNeuronCount.get(), this);

                    if(i == 0)    // input layer matrix
                        matrix = getRandomMatrix(hiddenLayerNeuronCount.get(), inputLayerNeuronCount.get());
                    else          // hidden layer matrices
                        matrix = getRandomMatrix(hiddenLayerNeuronCount.get(), hiddenLayerNeuronCount.get())

                    weightAggregators.get(i).setMatrix(matrix, this);
                }

                weightAggregators.get(aggregatorCount.get()).registerPersistent(numberOfClasses.get(), this);
                matrix = getRandomMatrix(numberOfClasses.get(), hiddenLayerNeuronCount.get());
                weightAggregators.get(aggregatorCount.get()).setMatrix(matrix, this);
            } catch (InstantiationException e) {
                e.printStackTrace();
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            }
        }


    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STATE_ID, IntOverwriteAggregator.class);
        registerPersistentAggregator(NUMBER_OF_CLASSES_ID, IntSumAggregator.class);
        registerPersistentAggregator(NUMBER_OF_INPUT_NEUTRONS_ID, IntSumAggregator.class);
        registerPersistentAggregator(NUMBER_OF_NETWORKS_ID, IntMaxAggregator.class);
        registerPersistentAggregator(NUMBER_OF_HIDDEN_LAYERS_ID, IntOverwriteAggregator.class);
    }

    private DoubleDenseMatrix getRandomMatrix(int rows, int cols) {
        DoubleDenseMatrix matrix = new DoubleDenseMatrix(rows, cols);

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                Double randVal = getRandomInRange(-EPSILON, EPSILON);
                matrix.set(i, j, Integer.MIN_VALUE);
            }
        }
        return matrix;
    }

    private double getRandomInRange(Double min, Double max) {
        Double rand = random.nextDouble();
        return min + (max - min) * rand;
    }
}

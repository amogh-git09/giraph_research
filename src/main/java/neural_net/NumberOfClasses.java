package neural_net;

import org.apache.giraph.aggregators.IntMaxAggregator;
import org.apache.giraph.aggregators.IntOverwriteAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVectorSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class NumberOfClasses extends DefaultMasterCompute{
    public static final int HIDDEN_LAYER_GENERATION = 0;

    public static final String STATE_ID = "StateAggregator";
    public static final String NUMBER_OF_CLASSES_ID = "NumberOfClassesAggregator";
    public static final String NUMBER_OF_INPUT_NEUTRONS_ID = "NumberOfInputNeutronsAggregator";
    public static final String NUMBER_OF_NETWORKS_ID = "NumberOfNetworksAggregator";

    @Override
    public void compute() {

    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STATE_ID, IntOverwriteAggregator.class);
        registerPersistentAggregator(NUMBER_OF_CLASSES_ID, IntSumAggregator.class);
        registerPersistentAggregator(NUMBER_OF_INPUT_NEUTRONS_ID, IntSumAggregator.class);
        registerPersistentAggregator(NUMBER_OF_NETWORKS_ID, IntMaxAggregator.class);
        registerPersistentAggregator("", IntDenseVectorSumAggregator.class);
    }
}

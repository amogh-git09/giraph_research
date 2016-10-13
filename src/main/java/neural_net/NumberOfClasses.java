package neural_net;

import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;

/**
 * Created by amogh-lab on 16/10/13.
 */
public class NumberOfClasses extends DefaultMasterCompute{
    public static final String ID = "NumberOfClassesAggregator";

    @Override
    public void compute() {
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(ID, IntSumAggregator.class);
    }
}

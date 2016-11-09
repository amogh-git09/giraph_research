package master_compute;

import org.apache.giraph.aggregators.IntOverwriteAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;

/**
 * Created by amogh-lab on 16/11/09.
 */
public class NNMasterCompute extends DefaultMasterCompute {
    //STAGES
    public static final int FRONT_EDGES_GENERATION_STAGE = 1;
    public static final int BACK_EDGES_GENERATION_STAGE = 2;
    public static final int DATA_LOAD_STAGE = 3;
    public static final int FORWARD_PROPAGATION_STAGE = 4;
    public static final int BACKWARD_PROPAGATION_STAGE = 5;

    public static final String STAGE_AGG_ID = "StageAggregator";

    @Override
    public void compute() {

    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STAGE_AGG_ID, IntSumAggregator.class);
    }
}

import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.hadoop.io.IntWritable;

/**
 * Created by amogh09 on 16/12/25.
 */
public class NNMasterCompute extends DefaultMasterCompute{
    public static final int HIDDEN_LAYER_GENERATION_STAGE = 0;
    public static final String STAGE_ID = "StageAggregator";

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STAGE_ID, IntSumAggregator.class);
        setAggregatedValue(STAGE_ID, new IntWritable(HIDDEN_LAYER_GENERATION_STAGE));
    }
}

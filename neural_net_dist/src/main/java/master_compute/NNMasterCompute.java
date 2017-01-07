package master_compute;

import config.Config;
import debug.Logger;
import debug.Timer;
import org.apache.giraph.aggregators.DoubleSumAggregator;
import org.apache.giraph.aggregators.IntSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;
import org.apache.hadoop.io.IntWritable;

/**
 * Created by amogh-lab on 16/11/09.
 */
public class NNMasterCompute extends DefaultMasterCompute {
    //STAGES
    public static final int STABILIZE_INITIAL_NETWORK = 1;
    public static final int FRONT_EDGES_GENERATION_STAGE = 2;
    public static final int BACK_EDGES_GENERATION_STAGE = 3;
    public static final int FORWARD_PROPAGATION_STAGE = 4;
    public static final int BACKWARD_PROPAGATION_STAGE = 5;

    public static final String STAGE_AGG_ID = "StageAggregator";
//    public static final String DATA_SET_INDEX_AGG = "DataSetIndex";
    public static final String COST_AGGREGATOR = "costAggregator";
    public static final String ITER_AGGREGATOR = "IterationAggregator";
    boolean startTimeRegistered = false;
    long startTime;
    int prevStage = 0;
    Timer timer = new Timer();

    @Override
    public void compute() {
        IntWritable iterations = getAggregatedValue(ITER_AGGREGATOR);

        if(getSuperstep() == 0) {
            Logger.i("Beginning computation with following network architecture");
            Config.printConfig();
        }

        if(iterations.get() > Config.MAX_ITER) {
            Logger.i("Halting");
            timer.report();
            haltComputation();
        }

        if(!startTimeRegistered) {
            startTime = System.currentTimeMillis();
            startTimeRegistered = true;
            Logger.p(String.format("Registered start time: %d", startTime));
        }

        IntWritable stage = getAggregatedValue(STAGE_AGG_ID);

        if (stage.get() == FORWARD_PROPAGATION_STAGE) {
            timer.start();
        }

        if(Logger.PERF) {
            if (prevStage != stage.get()) {
                long currTime = System.currentTimeMillis();
                Logger.p(String.format("PrevStage: %d, currStage: %d, diff = %.3f secs", prevStage, stage.get(),
                        (currTime - startTime) / (double) 1000));
                prevStage = stage.get();
                startTime = currTime;
            }
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerPersistentAggregator(STAGE_AGG_ID, IntSumAggregator.class);
//        registerPersistentAggregator(DATA_SET_INDEX_AGG, IntSumAggregator.class);
        registerPersistentAggregator(COST_AGGREGATOR, DoubleSumAggregator.class);
        registerPersistentAggregator(ITER_AGGREGATOR, IntSumAggregator.class);
//        setAggregatedValue(DATA_SET_INDEX_AGG, new IntWritable(1));
    }

    public static String idToStage(int id) {
        switch (id) {
            case FRONT_EDGES_GENERATION_STAGE:
                return "FRONT EDGES GENERATION STAGE";
            case BACK_EDGES_GENERATION_STAGE:
                return "BACK EDGES GENERATION STAGE";
//            case DATA_LOAD_STAGE:
//                return "DATA LOAD STAGE";
            case FORWARD_PROPAGATION_STAGE:
                return "FORWARD PROPAGATION STAGE";
            case BACKWARD_PROPAGATION_STAGE:
                return "BACKWARD PROPAGATION STAGE";
            case STABILIZE_INITIAL_NETWORK:
                return "STABILIZE INITIAL NETWORK";
        }

        return "UNKOWN STAGE";
    }
}

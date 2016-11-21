package neural_net;

import org.apache.giraph.worker.WorkerContext;

/**
 * Created by amogh09 on 16/11/21.
 */
public class PerformanceContext extends WorkerContext{

    boolean startTimeRegistered = false;
    long startTime;
    int prevStage = 0;

    @Override
    public void preApplication() throws InstantiationException, IllegalAccessException {

    }

    @Override
    public void postApplication() {

    }

    @Override
    public void preSuperstep() {

    }

    @Override
    public void postSuperstep() {

    }
}

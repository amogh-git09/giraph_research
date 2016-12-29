package neural_net;

import org.apache.giraph.worker.WorkerContext;

/**
 * Created by amogh09 on 16/11/21.
 */
public class PerformanceContext extends WorkerContext{

    boolean startTimeRegistered = false;
    long startTime;
    int prevStage = 0;
    Timer timer = new Timer();

    @Override
    public void preApplication() throws InstantiationException, IllegalAccessException {
        timer.start();
    }

    @Override
    public void postApplication() {
        timer.report();
    }

    @Override
    public void preSuperstep() {

    }

    @Override
    public void postSuperstep() {

    }
}

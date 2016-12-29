import org.apache.giraph.worker.WorkerContext;

/**
 * Created by amogh09 on 16/12/29.
 */
public class NNWorkerContext extends WorkerContext{
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

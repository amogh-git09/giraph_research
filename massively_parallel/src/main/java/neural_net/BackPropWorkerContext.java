package neural_net;

import org.apache.giraph.worker.WorkerContext;

/**
 * Created by amogh-lab on 16/11/07.
 */
public class BackPropWorkerContext extends WorkerContext {
    //abalone
    static int MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
    static int INPUT_LAYER_NEURON_COUNT = 7;
    static int HIDDEN_LAYER_NEURON_COUNT = 7;
    static int OUTPUT_LAYER_NEURON_COUNT = 29;

    static int BIAS_UNIT;
    static int INPUT_LAYER;
    static int OUTPUT_LAYER;
    static String DELIMITER;
    static double LEARNING_RATE;
    static int MAX_ITER = 1500;

    @Override
    public void preApplication() throws InstantiationException, IllegalAccessException {
        //abalone
        MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
        INPUT_LAYER_NEURON_COUNT = 7;
        HIDDEN_LAYER_NEURON_COUNT = 7;
        OUTPUT_LAYER_NEURON_COUNT = 29;

        //flower
//    MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
//    INPUT_LAYER_NEURON_COUNT = 4;
//    HIDDEN_LAYER_NEURON_COUNT = 4;
//    OUTPUT_LAYER_NEURON_COUNT = 3;

        //test
//    MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
//    INPUT_LAYER_NEURON_COUNT = 2;
//    HIDDEN_LAYER_NEURON_COUNT = 2;
//    OUTPUT_LAYER_NEURON_COUNT = 1;

        BIAS_UNIT = 0;
        INPUT_LAYER = 1;
        OUTPUT_LAYER = -1;
        DELIMITER = ":";
        LEARNING_RATE = 0.1;
        MAX_ITER = 1500;
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

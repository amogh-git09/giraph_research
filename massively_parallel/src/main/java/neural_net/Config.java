package neural_net;

/**
 * Created by amogh-lab on 16/11/16.
 */
public class Config {
    static final int BIAS_UNIT = 0;
    static final int INPUT_LAYER = 1;
    static final int OUTPUT_LAYER = -1;
    static final String DELIMITER = ":";
    static final double LEARNING_RATE = 0.1;
    static final int MAX_ITER = 1500;
    static final boolean TEST_WEIGHTS = false;

    //abalone
    static final int MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
    static final int INPUT_LAYER_NEURON_COUNT = 7;
    static final int HIDDEN_LAYER_NEURON_COUNT = 7;
    static final int OUTPUT_LAYER_NEURON_COUNT = 29;
}
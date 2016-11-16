package neural_net;

/**
 * Created by amogh-lab on 16/11/16.
 */
public class Config {
    static int BIAS_UNIT = 0;
    static int INPUT_LAYER = 1;
    static int OUTPUT_LAYER = -1;
    static String DELIMITER = ":";
    static double LEARNING_RATE = 0.1;
    static int MAX_ITER = 1500;

    //abalone
    static int MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2
    static int INPUT_LAYER_NEURON_COUNT = 7;
    static int HIDDEN_LAYER_NEURON_COUNT = 7;
    static int OUTPUT_LAYER_NEURON_COUNT = 29;
}

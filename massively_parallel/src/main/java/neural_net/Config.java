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
    static final int MAX_ITER = 50;
    static final int MAX_HIDDEN_LAYER_NUM = 2;                 // minimum value 2

    static final boolean TEST_WEIGHTS = true;

//    // MNIST
//    static final int INPUT_LAYER_NEURON_COUNT = 784;
//    static final int OUTPUT_LAYER_NEURON_COUNT = 10;
//    static final int[] LAYER_COUNTS = {INPUT_LAYER_NEURON_COUNT, 300, OUTPUT_LAYER_NEURON_COUNT};

//    //abalone
//    static final int INPUT_LAYER_NEURON_COUNT = 7;
//    static final int OUTPUT_LAYER_NEURON_COUNT = 29;
//    static final int[] LAYER_COUNTS = {INPUT_LAYER_NEURON_COUNT, 7, OUTPUT_LAYER_NEURON_COUNT};

//    //iris
//    static final int INPUT_LAYER_NEURON_COUNT = 4;
//    static final int OUTPUT_LAYER_NEURON_COUNT = 3;
//    static final int[] LAYER_COUNTS = {INPUT_LAYER_NEURON_COUNT, 3, OUTPUT_LAYER_NEURON_COUNT};

//    test
    static final int INPUT_LAYER_NEURON_COUNT = 2;
    static final int OUTPUT_LAYER_NEURON_COUNT = 1;
    static final int[] LAYER_COUNTS = {INPUT_LAYER_NEURON_COUNT, 2, OUTPUT_LAYER_NEURON_COUNT};
}

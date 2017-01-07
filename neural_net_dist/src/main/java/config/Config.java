package config;

import debug.Logger;

/**
 * Created by amogh09 on 16/12/15.
 */
public class Config {
    public static final int DATA_SIZE = 1;
    public static final double LEARNING_RATE = 0.05;

    public static final int INPUT_LAYER_NEURON_COUNT = 5000;
    public static final int OUTPUT_LAYER_NEURON_COUNT = 10;
    public static final int[] LAYER_TO_NEURON = {
            INPUT_LAYER_NEURON_COUNT, 3000, OUTPUT_LAYER_NEURON_COUNT};

//    public static final int INPUT_LAYER_NEURON_COUNT = 2;
//    public static final int OUTPUT_LAYER_NEURON_COUNT = 1;
//    public static final int[] LAYER_TO_NEURON = {
//            INPUT_LAYER_NEURON_COUNT, 2, OUTPUT_LAYER_NEURON_COUNT};

    public static final int MAX_HIDDEN_LAYER_NUM = 2;
    public static final int INPUT_LAYER = 1;
    public static final int OUTPUT_LAYER = -1;
    public static final int BIAS_UNIT = 0;

    public static final boolean TESTING = true;
    public static final String DELIMITER = ":";
    public static final int MAX_ITER = 2;

    public static void printConfig() {
        String msg = "";
        for(int i = 0; i < LAYER_TO_NEURON.length; i++) {
            msg = msg + LAYER_TO_NEURON[i] +  " : ";
        }

        Logger.i(msg);
    }
}

import org.apache.hadoop.io.Text;

/**
 * Created by amogh09 on 16/12/24.
 */
public class Config {
    public static final String DELIMITER = ":";
    public static final double EPSILON = 0.5;
    public static final double LEARNING_RATE = 0.05;
    public static int MAX_ITER = 10;

    public static int dataSize = 0;
    public static int checker = 0;

    //abalone
//    public static int INPUT_LAYER_NEURON_COUNT = 8;     //including bias
//    public static final int OUTPUT_LAYER_NEURON_COUNT = 29;
//    public static final int[] ARCHITECTURE = {
//            INPUT_LAYER_NEURON_COUNT, 7, OUTPUT_LAYER_NEURON_COUNT};

    //iris
//    public static int INPUT_LAYER_NEURON_COUNT = 5;     //including bias
//    public static final int OUTPUT_LAYER_NEURON_COUNT = 3;
//    public static final int[] ARCHITECTURE = {
//            INPUT_LAYER_NEURON_COUNT, 5, 5, OUTPUT_LAYER_NEURON_COUNT};

    //mnist
    public static int INPUT_LAYER_NEURON_COUNT = 785;     //including bias
    public static final int OUTPUT_LAYER_NEURON_COUNT = 10;
    public static final int[] ARCHITECTURE = {
            INPUT_LAYER_NEURON_COUNT, 300, OUTPUT_LAYER_NEURON_COUNT};


    public static final boolean TESTING = false;

    public static final int INPUT = 0;
    public static final int OUTPUT = ARCHITECTURE.length - 1;
    public static final int HIDDEN_LAYER_COUNT = ARCHITECTURE.length - 2;

    public static Text getVertexId(int dataNum, int layerNum) {
        return new Text(dataNum + DELIMITER + layerNum);
    }

    public static int getLayerNum(Text vertexId) {
        String[] tokens = vertexId.toString().split(Config.DELIMITER);
        return Integer.parseInt(tokens[1]);
    }

    public static int getDataNum(Text vertexId) {
        String[] tokens = vertexId.toString().split(Config.DELIMITER);
        return Integer.parseInt(tokens[0]);
    }
}

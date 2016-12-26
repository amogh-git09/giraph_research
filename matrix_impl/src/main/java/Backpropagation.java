import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Vector;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;
import java.util.Random;

/**
 * Created by amogh09 on 16/12/25.
 */
public class Backpropagation extends BasicComputation<Text, NeuronValue,
        NullWritable, DenseVectorWritable>{

    @Override
    public void compute(Vertex<Text, NeuronValue, NullWritable> vertex,
                        Iterable<DenseVectorWritable> messages) throws IOException {

        int dataNum = Config.getDataNum(vertex.getId());
        int layerNum = Config.getLayerNum(vertex.getId());
        IntWritable stage = getAggregatedValue(NNMasterCompute.STAGE_ID);
        Logger.d("\n\nVertex Id: " + vertex.getId() + ",  SS: " + getSuperstep());
        Logger.d("Stage: " + stage.get());

        switch (stage.get()) {
            case NNMasterCompute.HIDDEN_LAYER_GENERATION_STAGE:
                switch (layerNum) {
                    case Config.INPUT:
                        for(int i=1; i<=Config.HIDDEN_LAYER_COUNT; i++) {
                            generateHiddenLayer(dataNum, i);
                        }
                        break;

                    case Config.OUTPUT:
                        if(dataNum == 1) {
                            aggregate(NNMasterCompute.STAGE_ID, new IntWritable(1));
                        }
                        vertex.voteToHalt();
                        break;
                }
                break;

            case NNMasterCompute.FORWARD_PROPAGATION_STAGE:
                switch (layerNum) {
                    case Config.INPUT:
                        forwardPropActivations(vertex, dataNum, layerNum);
                        vertex.voteToHalt();
                        break;

                    case Config.OUTPUT:
                        for(DenseVectorWritable v : messages) {
                            printVector(v.vector);
                        }

                        vertex.voteToHalt();
                        break;

                    default:
                        for(DenseVectorWritable v : messages) {
                            printVector(v.vector);
                            updateActivations(vertex, messages, layerNum);
                            forwardPropActivations(vertex, dataNum, layerNum);
                        }

                        vertex.voteToHalt();
                        break;
                }
                break;
        }

        Logger.d("\n\n");
    }

    private boolean updateActivations(Vertex<Text, NeuronValue, NullWritable> vertex,
                                   Iterable<DenseVectorWritable> messages, int layerNum) {

        DenseVector input = null;
        DenseMatrix theta = vertex.getValue().getWeights().getMatrix();
        int checker = 0;

        for (DenseVectorWritable m : messages) {
            input = m.vector;
            checker++;
        }

        if(checker == 1) {
            Vector activations = new DenseVector(Config.ARCHITECTURE[layerNum]);
            activations = theta.mult(input, activations);
            vertex.getValue().setActivations(new DenseVectorWritable((DenseVector) activations));
            Logger.d("Updated activations:");
            printVector(vertex.getValue().getActivations().vector);
            return true;
        } else if (checker == 0) {
            return false;
        } else {
            throw new IllegalStateException("More than 1 message received");
        }
    }

    private void forwardPropActivations(Vertex<Text, NeuronValue, NullWritable> vertex, int dataNum,
                                        int layerNum) {

        Text dstId = Config.getVertexId(dataNum, getNextLayerNum(layerNum));
        sendMessage(dstId, vertex.getValue().getActivations());
        Logger.d("forward propagating vector to " + dstId.toString());
    }

    private void generateHiddenLayer(int dataNum, int layerNum) throws IOException {
        DenseMatrixWritable matrix = getAggregatedValue(NNMasterCompute.getWeightAggregatorName(layerNum));
        NeuronValue val = new NeuronValue(null, matrix, null);
        Text id = Config.getVertexId(dataNum, layerNum);
        addVertexRequest(id, val);
        Logger.d("adding vertex: " + id.toString());
        Logger.d("weights: ");
        printMatrix(matrix.getMatrix());
    }

    public static void printMatrix(DenseMatrix m) {
        if(Logger.DEBUG) {
            for (int i = 0; i < m.numRows(); i++) {
                StringBuilder str = new StringBuilder();
                for (int j = 0; j < m.numColumns(); j++) {
                    str.append(m.get(i, j) + "  ");
                }
                Logger.d(str.toString());
            }
        }
    }

    private void printVector(DenseVector v) {
        if(Logger.DEBUG) {
            StringBuilder str = new StringBuilder();
            for (int i = 0; i < v.size(); i++) {
                str.append(v.get(i) + "  ");
            }
            Logger.d(str.toString());
        }
    }

    public static int getNextLayerNum(int layerNum) {
        switch (layerNum) {
            case Config.OUTPUT:
                return Config.INPUT;
            case Config.INPUT + Config.HIDDEN_LAYER_COUNT:
                return Config.OUTPUT;
            default:
                return layerNum + 1;
        }
    }

    public static int getPrevLayerNum(int layerNum) {
        switch (layerNum) {
            case Config.INPUT:
                return Config.OUTPUT;
            case Config.OUTPUT:
                return Config.INPUT + Config.HIDDEN_LAYER_COUNT;
            default:
                return layerNum - 1;
        }
    }

    private static DenseMatrix generateRandomMatrix(int rows, int cols) {
        double min = - Config.EPSILON;
        double max = Config.EPSILON;
        Random r = new Random();
        double[][] data = new double[rows][cols];

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                data[i][j] = min + (max - min) * r.nextDouble();
            }
        }

        return new DenseMatrix(data);
    }

    public static void printActivations(Vertex<Text, NeuronValue, NullWritable> vertex) {
        DenseVector vec = vertex.getValue().getActivations().vector;
        StringBuilder stringBuilder = new StringBuilder();
        for(int i=0; i<vec.size(); i++) {
            stringBuilder.append(vec.get(i) + "  ");
        }
        stringBuilder.append("\n");

        Logger.d(stringBuilder.toString());
    }

    public static void printOutput(Vertex<Text, NeuronValue, NullWritable> vertex) {
        DenseVector vec = vertex.getValue().getOutput().vector;
        StringBuilder stringBuilder = new StringBuilder();
        for(int i=0; i<vec.size(); i++) {
            stringBuilder.append(vec.get(i) + "  ");
        }
        stringBuilder.append("\n");

        Logger.d(stringBuilder.toString());
    }
}

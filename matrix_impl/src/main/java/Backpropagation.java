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

                    // OUTPUT LAYER
                    default:
                        if(dataNum == 1) {
                            aggregate(NNMasterCompute.STAGE_ID, new IntWritable(1));
                        }
                        vertex.voteToHalt();
                        break;
                }
                break;

            case NNMasterCompute.FORWARD_PROPAGATION_STAGE:
                if (layerNum == Config.OUTPUT) {
                    int checker = 0;
                    for(DenseVectorWritable v : messages) {
                        printVector(v.vector);
                        vertex.getValue().setActivations(v);
                        checker++;
                    }

                    if(checker != 1) {
                        throw new IllegalStateException("More than one messages received");
                    }

                    // switch stage
                    if(dataNum == 1) {
                        aggregate(NNMasterCompute.STAGE_ID, new IntWritable(1));
                    }
                } else {
                    DenseVector updatedActivations;
                    switch (layerNum) {
                        case Config.INPUT:
                            updatedActivations = getUpdatedActivations(vertex, messages, layerNum);
                            forwardPropActivations(updatedActivations, dataNum, layerNum);
                            vertex.voteToHalt();
                            break;

                        default:
                            for(DenseVectorWritable v : messages) {
                                printVector(v.vector);
                                vertex.getValue().setActivations(v);
                                updatedActivations = getUpdatedActivations(vertex, messages, layerNum);
                                forwardPropActivations(updatedActivations, dataNum, layerNum);
                            }

                            vertex.voteToHalt();
                            break;
                    }
                }
                break;

            case NNMasterCompute.BACKWARD_PROPAGATION_STAGE:
                if (layerNum == Config.OUTPUT) {
                    DenseVector delta = calculateErrors(vertex, layerNum);
                    Logger.d("delta:");
                    printVector(delta);
                    vertex.voteToHalt();
                }
                break;
        }

        Logger.d("\n\n");
    }

    private DenseVector calculateErrors(Vertex<Text, NeuronValue, NullWritable> vertex,
                                        int layerNum) {

        Vector delta = new DenseVector(Config.ARCHITECTURE[layerNum]);

        if(layerNum == Config.OUTPUT) {
            NeuronValue val = vertex.getValue();
            Vector activations = new DenseVector(val.getActivations().vector);
            delta = activations.add(-1d, (Vector) val.getOutput().vector);
        }

        return (DenseVector) delta;
    }

    private DenseVector getUpdatedActivations(Vertex<Text, NeuronValue, NullWritable> vertex,
                                              Iterable<DenseVectorWritable> messages, int layerNum) {

        DenseVector input = null;
//        DenseMatrix theta = vertex.getValue().getWeights().getMatrix();

        String aggName = NNMasterCompute.getWeightAggregatorName(layerNum);
        DenseMatrixWritable thetaWr = getAggregatedValue(aggName);
        DenseMatrix theta = thetaWr.getMatrix();

        int checker = 0;

        for (DenseVectorWritable m : messages) {
            input = m.vector;
            checker++;
        }

        if(checker == 1) {
            Vector activations = new DenseVector(Config.ARCHITECTURE[getNextLayerNum(layerNum)]);
            activations = theta.mult(input, activations);
            sigmoid(activations);
            Logger.d("Updated activations:");
            printVector(activations);
            return (DenseVector) activations;
        } else if (checker == 0) {
            //input layer
            Vector activations = new DenseVector(Config.ARCHITECTURE[getNextLayerNum(layerNum)]);
            activations = theta.mult(vertex.getValue().getActivations().vector, activations);
            sigmoid(activations);
            Logger.d("Updated activations:");
            printVector(activations);
            return (DenseVector) activations;
        } else {
            throw new IllegalStateException("More than 1 message received");
        }
    }

    private void forwardPropActivations(DenseVector activations, int dataNum,
                                        int layerNum) {

        Text dstId = Config.getVertexId(dataNum, getNextLayerNum(layerNum));
        sendMessage(dstId, new DenseVectorWritable(activations));
        Logger.d("forward propagating vector to " + dstId.toString());
    }

    private void generateHiddenLayer(int dataNum, int layerNum) throws IOException {
//        DenseMatrixWritable matrix = getAggregatedValue(NNMasterCompute.getWeightAggregatorName(layerNum));
//        NeuronValue val = new NeuronValue(null, matrix, null);

        NeuronValue val = new NeuronValue(null, null, null);

        Text id = Config.getVertexId(dataNum, layerNum);
        addVertexRequest(id, val);
        Logger.d("adding vertex: " + id.toString());
        Logger.d("weights: ");
//        printMatrix(matrix.getMatrix());
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

    private void printVector(Vector v) {
        if(Logger.DEBUG) {
            StringBuilder str = new StringBuilder();
            for (int i = 0; i < v.size(); i++) {
                str.append(v.get(i) + "  ");
            }
            Logger.d(str.toString());
        }
    }

    public static int getNextLayerNum(int layerNum) {
        if(layerNum == Config.OUTPUT)
            return Config.INPUT;
        else if (layerNum == Config.INPUT + Config.HIDDEN_LAYER_COUNT)
            return Config.OUTPUT;
        else
            return layerNum + 1;
    }

    public static int getPrevLayerNum(int layerNum) {
        if(layerNum == Config.INPUT)
            return Config.OUTPUT;
        else if(layerNum == Config.OUTPUT)
            return Config.INPUT + Config.HIDDEN_LAYER_COUNT;
        else
            return layerNum - 1;
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

    private static void sigmoid(Vector vector) {
        for(int i = 0; i < vector.size(); i++) {
            double value = vector.get(i);
            vector.set(i, sigmoid(value));
        }
    }

    private static double sigmoid(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }
}

import no.uib.cipr.matrix.*;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.python.antlr.op.Del;

import java.io.IOException;
import java.util.Iterator;
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

                        aggregate(NNMasterCompute.DATANUM_ID, new IntWritable(1));
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

                    Logger.i("Cost = " +
                            (((DoubleWritable) getAggregatedValue(NNMasterCompute.COST_ID)).get()));

                    //find cost
                    double cost = cost(vertex);
                    aggregate(NNMasterCompute.COST_ID, new DoubleWritable(
                            ((DoubleWritable) getAggregatedValue(NNMasterCompute.COST_ID)).get()));
                    aggregate(NNMasterCompute.COST_ID, new DoubleWritable(cost));

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
                if (layerNum != Config.INPUT) {
                    DenseVector delta = calculateErrors(vertex, messages, layerNum);
                    Logger.d("delta:");
                    printVector(delta);

                    if(!isLastHiddenLayer(layerNum)) {
                        String aggName = NNMasterCompute.getDeltaAggregatorName(getNextLayerNum(layerNum));
                        Logger.d(String.format("%s", aggName));
                        DenseMatrixWritable DeltaWr = getAggregatedValue(aggName);
                        printMatrix(DeltaWr.getMatrix());
                    }

                    backpropError(dataNum, layerNum, delta);
                    vertex.voteToHalt();
                } else {
                    DenseMatrixWritable DeltaWr = getAggregatedValue(NNMasterCompute.getDeltaAggregatorName(getNextLayerNum(layerNum)));
                    Logger.d("Delta for layer: " + getNextLayerNum(layerNum));
                    printMatrix(DeltaWr.getMatrix());

                    aggregate(NNMasterCompute.STAGE_ID, new IntWritable(-stage.get()));
                    aggregate(NNMasterCompute.STAGE_ID, new IntWritable(NNMasterCompute.FORWARD_PROPAGATION_STAGE));
                }
                break;

            case NNMasterCompute.WEIGHT_UPDATE_STAGE:
                if (layerNum == Config.OUTPUT) {

                } else {
                    IntWritable m = getAggregatedValue(NNMasterCompute.DATANUM_ID);
                    String aggName = NNMasterCompute.getDeltaAggregatorName(layerNum);
                    DenseMatrixWritable Delta = getAggregatedValue(aggName);
                    DenseMatrix gradients = new DenseMatrix(Delta.getNumRows(), Delta.getNumCols());
                    gradients = Delta.multAdd((1d/m.get(), Delta, )
                }
                break;
        }
    }

    private double cost(Vertex<Text, NeuronValue, NullWritable> vertex) {
        DenseVector output = vertex.getValue().getOutput().vector;
        DenseVector activations = vertex.getValue().getActivations().vector;

        double cost = 0;

        for(Iterator<VectorEntry> i = output.iterator(); i.hasNext(); ) {
            VectorEntry element = i.next();
            double left = element.get() * Math.log(activations.get(element.index()));
            double right = (1 - element.get()) * Math.log(1 - activations.get(element.index()));
            cost += - left - right;
        }

        return cost;
    }

    private void backpropError(int dataNum, int currentLayerNum, DenseVector delta) {
        int targetLayerNum = getPrevLayerNum(currentLayerNum);
        Text dstId = Config.getVertexId(dataNum, targetLayerNum);
        sendMessage(dstId, new DenseVectorWritable(delta));
        Logger.d(String.format("Sending the msg below to %s", dstId.toString()));
        printVector(delta);
    }

    private DenseVector calculateErrors(Vertex<Text, NeuronValue, NullWritable> vertex,
                                        Iterable<DenseVectorWritable> messages,
                                        int layerNum) {

        Vector delta = new DenseVector(getNeuronCount(layerNum));

        if(layerNum == Config.OUTPUT) {
            NeuronValue val = vertex.getValue();
            Vector activations = new DenseVector(val.getActivations().vector);
            Logger.d("Subtracting:");
            printVector(activations);
            Logger.d("----------------");
            printVector(val.getOutput().vector);
            delta = activations.add(-1d, (Vector) val.getOutput().vector);
        } else {
            int checker = 0;
            for(DenseVectorWritable v : messages) {
                String aggName = NNMasterCompute.getWeightAggregatorName(layerNum);
                DenseMatrixWritable theta = getAggregatedValue(aggName);
                delta = theta.getMatrix().transMult(v.vector, delta);

                DenseVector activations = vertex.getValue().getActivations().vector;
                for(Iterator<VectorEntry> i = delta.iterator(); i.hasNext(); ) {
                    VectorEntry elem = i.next();
                    double activation = activations.get(elem.index());
                    elem.set(elem.get() * activation * (1 - activation));
                }

                // Delta += activations * incoming_delta
                Matrix deltaM = new DenseMatrix(v.vector);
                Matrix activationsM = new DenseMatrix(activations);
                Matrix Delta = new DenseMatrix(deltaM.numRows(), activationsM.numRows());
                Logger.d("Multiplying:");
                printMatrix(deltaM);
                Logger.d("with transpose of:");
                printMatrix(activationsM);
                Delta = deltaM.transBmult(activationsM, Delta);
                Logger.d("Result:");
                printMatrix(Delta);

                String deltaAggName = NNMasterCompute.getDeltaAggregatorName(layerNum);
                Logger.d("Aggregating result to " + deltaAggName);
                aggregate(deltaAggName, new DenseMatrixWritable((DenseMatrix) Delta));

                checker++;
            }

            if(checker != 1) {
                throw new IllegalStateException(String.format("Received %d messages, expected %d", checker, 1));
            }
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
            int nextLayerNeuronCount = getNextLayerNeuronCount(layerNum);

            Vector activations = new DenseVector(nextLayerNeuronCount);
            Logger.d("Multiplying:");
            printMatrix(theta);
            Logger.d("-----------");
            printVector(input);
            activations = theta.mult(input, activations);

            Logger.d("Updated activations:");
            printVector(activations);

            sigmoid(activations);
            Logger.d("After sigmoid:");

            if(!isLastHiddenLayer(layerNum))
                activations.set(0, 1d);

            printVector(activations);
            return (DenseVector) activations;
        } else if (checker == 0) {
            //input layer
            Vector activations = new DenseVector(getNextLayerNeuronCount(layerNum));
            input = vertex.getValue().getActivations().vector;

            Logger.d("Multiplying:");
            printMatrix(theta);
            Logger.d("-----------");
            printVector(input);

            activations = theta.mult(input, activations);
            Logger.d("Updated activations:");
            printVector(activations);

            sigmoid(activations);
            Logger.d("After sigmoid:");

            activations.set(0, 1d);
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

    public static void printMatrix(Matrix m) {
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

    public static int getNeuronCount(int layerNum) {
        return Config.ARCHITECTURE[layerNum];
    }

    public static int getNextLayerNeuronCount(int layerNum) {
        return Config.ARCHITECTURE[getNextLayerNum(layerNum)];
    }

    public static boolean isLastHiddenLayer(int layerNum) {
        return getNextLayerNum(layerNum) == Config.OUTPUT;
    }

    public void turnLayerToActive(int layerNum) {
        IntWritable dataNumWr = getAggregatedValue(NNMasterCompute.DATANUM_ID);
        for(int i=1; i<=dataNumWr.get(); i++) {
            Text dstId = Config.getVertexId(i, layerNum);
            sendMessage(dstId, new DenseVectorWritable());
        }
    }
}

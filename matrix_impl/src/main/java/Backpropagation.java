import no.uib.cipr.matrix.DenseVector;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Created by amogh09 on 16/12/25.
 */
public class Backpropagation extends BasicComputation<Text, NeuronValue,
        NullWritable, DenseVectorWritable>{

    @Override
    public void compute(Vertex<Text, NeuronValue, NullWritable> vertex,
                        Iterable<DenseVectorWritable> messages) throws IOException {

        if(getSuperstep() == 0) {
            Logger.d("Hello from vertex: " + vertex.getId());
            int layerNum = Config.getLayerNum(vertex.getId());
            if(layerNum == Config.INPUT) {
                Logger.d("activations are:");
                printActivations(vertex);
            } else {
                Logger.d("Outputs are");
                printOutput(vertex);
            }
        } else {
            vertex.voteToHalt();
        }

        int dataNum = Config.getDataNum(vertex.getId());
        int layerNum = Config.getLayerNum(vertex.getId());
        IntWritable stage = getAggregatedValue(NNMasterCompute.STAGE_ID);

        switch (stage.get()) {
            case NNMasterCompute.HIDDEN_LAYER_GENERATION_STAGE:
                break;

            switch (layerNum) {
                case Config.INPUT:

                    break;

                case Config.OUTPUT:
                    break;
            }
        }
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

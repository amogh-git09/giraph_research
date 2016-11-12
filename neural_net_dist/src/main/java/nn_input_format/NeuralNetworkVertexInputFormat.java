package nn_input_format;

import debug.Logger;
import distributed_net.DistributedNeuralNetwork;
import distributed_net.NeuronValue;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.VertexInputFormat;
import org.apache.giraph.io.VertexReader;
import org.apache.giraph.io.formats.GiraphTextInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

import java.io.IOException;
import java.util.List;

/**
 * Created by amogh-lab on 16/10/10.
 */
public class NeuralNetworkVertexInputFormat extends VertexInputFormat<Text, NeuronValue, DoubleWritable> {
    GiraphTextInputFormat textInputFormat = new GiraphTextInputFormat();
    static final int INPUT_LAYER = 1;

    @Override
    public VertexReader<Text, NeuronValue, DoubleWritable> createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return new NeuralNetworkVertexReader();
    }

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public List<InputSplit> getSplits(JobContext context, int minSplitCountHint) throws IOException, InterruptedException {
        return textInputFormat.getVertexSplits(context);
    }

    public class NeuralNetworkVertexReader extends VertexReader<Text, NeuronValue, DoubleWritable> {
        private RecordReader<LongWritable, Text> lineRecordReader;
        Vertex<Text, NeuronValue, DoubleWritable> currentVertex;

        @Override
        public void initialize(InputSplit inputSplit, TaskAttemptContext context) throws IOException, InterruptedException {
            lineRecordReader = new LineRecordReader();
            lineRecordReader.initialize(inputSplit, context);
        }

        @Override
        public boolean nextVertex() throws IOException, InterruptedException {
            if(!lineRecordReader.nextKeyValue()) {
                return false;
            }

            Text line = lineRecordReader.getCurrentValue();
            String data = line.toString();

            Vertex<Text, NeuronValue, DoubleWritable> vertex = getConf().createVertex();
            Text id = new Text(1 + DistributedNeuralNetwork.DELIMITER + 0);
            int nextLayerNeuronCount = DistributedNeuralNetwork.getNeuronCount(
                    DistributedNeuralNetwork.getNextLayerNum(DistributedNeuralNetwork.INPUT_LAYER));
            Logger.d("First neuron, derivatives size = " + nextLayerNeuronCount);
            NeuronValue val = new NeuronValue(Double.parseDouble(data), 0d, 0d, 0, nextLayerNeuronCount);

            vertex.initialize(id, val);
            currentVertex = vertex;
            return true;
        }

        @Override
        public Vertex<Text, NeuronValue, DoubleWritable> getCurrentVertex() throws IOException, InterruptedException {
            Logger.d("Returning initialized vertex, derivatives size = " + currentVertex.getValue().getDerivativesLength());
            return currentVertex;
        }

        @Override
        public void close() throws IOException {
            lineRecordReader.close();
        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            return lineRecordReader.getProgress();
        }
    }
}

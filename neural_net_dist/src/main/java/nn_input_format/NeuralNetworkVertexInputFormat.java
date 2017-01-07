package nn_input_format;

import config.Config;
import debug.Logger;
import distributed_net.DistNeuralNet;
import distributed_net.EdgeValue;
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
public class NeuralNetworkVertexInputFormat extends VertexInputFormat<Text, NeuronValue, EdgeValue> {
    GiraphTextInputFormat textInputFormat = new GiraphTextInputFormat();

    @Override
    public VertexReader<Text, NeuronValue, EdgeValue> createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return new NeuralNetworkVertexReader();
    }

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public List<InputSplit> getSplits(JobContext context, int minSplitCountHint) throws IOException, InterruptedException {
        return textInputFormat.getVertexSplits(context);
    }

    public class NeuralNetworkVertexReader extends VertexReader<Text, NeuronValue, EdgeValue> {
        private RecordReader<LongWritable, Text> lineRecordReader;
        Vertex<Text, NeuronValue, EdgeValue> currentVertex;

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

            Vertex<Text, NeuronValue, EdgeValue> vertex = getConf().createVertex();
            Text id = DistNeuralNet.getNeuronId(Config.INPUT_LAYER, Config.BIAS_UNIT);
            int nextLayerNeuronCount = DistNeuralNet.getNextLayerNeuronCount(Config.INPUT_LAYER);
            Logger.d("First neuron, derivatives size = " + nextLayerNeuronCount);
            NeuronValue val = new NeuronValue(1d, 0d, 0d, 0, nextLayerNeuronCount);

            vertex.initialize(id, val);
            currentVertex = vertex;
            return true;
        }

        @Override
        public Vertex<Text, NeuronValue, EdgeValue> getCurrentVertex() throws IOException, InterruptedException {
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

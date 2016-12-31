import no.uib.cipr.matrix.DenseVector;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.VertexInputFormat;
import org.apache.giraph.io.VertexReader;
import org.apache.giraph.io.formats.GiraphTextInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

import java.io.IOException;
import java.util.List;

/**
 * Created by amogh09 on 16/12/19.
 */
public class NeuralNetworkVectorVertexInputFormat extends VertexInputFormat<Text, NeuronValue, NullWritable>{
    GiraphTextInputFormat inputFormat = new GiraphTextInputFormat();

    @Override
    public VertexReader<Text, NeuronValue, NullWritable> createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return new NeuralNetworkVertexReader();
    }

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public List<InputSplit> getSplits(JobContext context, int minSplitCountHint) throws IOException, InterruptedException {
        return inputFormat.getVertexSplits(context);
    }

    public class NeuralNetworkVertexReader extends VertexReader<Text, NeuronValue, NullWritable> {
        private TaskAttemptContext context;
        private RecordReader<LongWritable, Text> lineRecordReader;
        private int layer = 0;  // 0 for input, 1 for output
        private int dataNum = 0;
        private int classNum = -1;
        private Vertex<Text, NeuronValue, NullWritable> currentVertex;

        @Override
        public void initialize(InputSplit inputSplit, TaskAttemptContext context) throws IOException, InterruptedException {
            this.context = context;
            lineRecordReader = new LineRecordReader();
            lineRecordReader.initialize(inputSplit, context);
        }

        @Override
        public boolean nextVertex() throws IOException, InterruptedException {
            if (layer == Config.INPUT) {
                if (!lineRecordReader.nextKeyValue()) {
                    return false;
                }

                Text line = lineRecordReader.getCurrentValue();
                Logger.d("Reading data: " + line.toString());
                dataNum += 1;
                String[] tokens = line.toString().split(",");
                int len = tokens.length;
                Config.INPUT_LAYER_NEURON_COUNT = len;

                if(len != Config.INPUT_LAYER_NEURON_COUNT) {
                    throw new IllegalArgumentException(String.format("Expected %d features, found %d",
                            Config.INPUT_LAYER_NEURON_COUNT, len));
                }

                classNum = Integer.parseInt(tokens[len - 1]);
                Text id = Config.getVertexId(dataNum, layer);

                double[] data = new double[len];
                data[0] = 1d;
                for(int i=1; i<len; i++) {
                    data[i] = Double.parseDouble(tokens[i-1]);
                }
                DenseVectorWritable vec = new DenseVectorWritable(new DenseVector(data));
                NeuronValue val = new NeuronValue(vec, null);
                Vertex<Text, NeuronValue, NullWritable> vertex = getConf().createVertex();
                vertex.initialize(id, val);

                layer = Config.OUTPUT;
                currentVertex = vertex;

                Config.dataSize++;
                return true;
            } else {
                double[] data = new double[Config.OUTPUT_LAYER_NEURON_COUNT];

                for(int i=0; i<Config.OUTPUT_LAYER_NEURON_COUNT; i++) {
                    if(data.length == 1)
                        data[i] = i == classNum ? 0d : 1d;
                    else
                        data[i] = i == classNum ? 1d : 0d;
                }

                Text id = new Text(Config.getVertexId(dataNum, Config.OUTPUT));
                DenseVectorWritable vec = new DenseVectorWritable(new DenseVector(data));
                NeuronValue val = new NeuronValue(null, vec);
                Vertex<Text, NeuronValue, NullWritable> vertex = getConf().createVertex();
                vertex.initialize(id, val);

                layer = Config.INPUT;
                currentVertex = vertex;
                return true;
            }
        }

        @Override
        public Vertex<Text, NeuronValue, NullWritable> getCurrentVertex() throws IOException, InterruptedException {
            return currentVertex;
        }

        @Override
        public void close() throws IOException {

        }

        @Override
        public float getProgress() throws IOException, InterruptedException {
            return 0;
        }
    }
}

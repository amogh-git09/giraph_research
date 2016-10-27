package neural_net;

import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.VertexInputFormat;
import org.apache.giraph.io.VertexReader;
import org.apache.giraph.io.formats.GiraphTextInputFormat;
import org.apache.giraph.io.formats.TextVertexInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.LineRecordReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by amogh-lab on 16/10/10.
 */
public class NeuralNetworkVertexInputFormat extends VertexInputFormat<Text, NeuronValue, DoubleWritable> {
    GiraphTextInputFormat textInputFormat = new GiraphTextInputFormat();
    static final int OUTPUT_LAYER = -1;
    static final int INPUT_LAYER = 1;
    private int networkNum = 1;
    private int vertexNum = 1;
    private int layerNum = 1;
    private int counter = 0;

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
        private TaskAttemptContext context;

        @Override
        public void initialize(InputSplit inputSplit, TaskAttemptContext context) throws IOException, InterruptedException {
            this.context = context;
            lineRecordReader = new LineRecordReader();
            lineRecordReader.initialize(inputSplit, context);
        }

        @Override
        public boolean nextVertex() throws IOException, InterruptedException {
            return lineRecordReader.nextKeyValue();
        }

        @Override
        public Vertex<Text, NeuronValue, DoubleWritable> getCurrentVertex() throws IOException, InterruptedException {
            counter++;

            if(counter%1000 == 0) {
                System.out.println("Read " + counter + " vertices");
            }

            Text line = lineRecordReader.getCurrentValue();

            while(line.toString().equals("output")) {
                vertexNum = 1;
                lineRecordReader.nextKeyValue();
                line = lineRecordReader.getCurrentValue();
                layerNum = OUTPUT_LAYER;      //output layer
            }

            while (line.toString().equals("done")) {
                networkNum++;
                vertexNum = 1;
                layerNum = INPUT_LAYER;
                lineRecordReader.nextKeyValue();
                line = lineRecordReader.getCurrentValue();
            }
            String data = line.toString();

            Vertex<Text, NeuronValue, DoubleWritable> vertex = getConf().createVertex();
            Text id = new Text(networkNum + ":" + layerNum + ":" + vertexNum);
            vertexNum++;
            NeuronValue val;

            if(layerNum == OUTPUT_LAYER) {           // in case of output layer, store true/false value in 'error'
                val = new NeuronValue(0d, 0d, 0d, Integer.parseInt(data));
            }
            else
                val = new NeuronValue(Double.parseDouble(data), 0d, 0d, 0);

            vertex.initialize(id, val);

            return vertex;
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

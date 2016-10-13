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
public class NeuralNetworkVertexInputFormat extends VertexInputFormat<Text, DoubleWritable, DoubleWritable> {
    GiraphTextInputFormat textInputFormat = new GiraphTextInputFormat();

    @Override
    public VertexReader<Text, DoubleWritable, DoubleWritable> createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return new NeuralNetworkVertexReader();
    }

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public List<InputSplit> getSplits(JobContext context, int minSplitCountHint) throws IOException, InterruptedException {
        return textInputFormat.getVertexSplits(context);
    }

    public class NeuralNetworkVertexReader extends VertexReader<Text, DoubleWritable, DoubleWritable> {
        private RecordReader<LongWritable, Text> lineRecordReader;
        private TaskAttemptContext context;
        private static final int OUTPUT_LAYER = -1;
        private static final int INPUT_LAYER = 1;
        private int networkNum = 1;
        private int vertexNum = 1;
        private int layerNum = 1;

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
        public Vertex<Text, DoubleWritable, DoubleWritable> getCurrentVertex() throws IOException, InterruptedException {
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

            Vertex<Text, DoubleWritable, DoubleWritable> vertex = getConf().createVertex();
            Text id = new Text(networkNum + ":" + layerNum + ":" + vertexNum);
            vertexNum++;
            DoubleWritable val = new DoubleWritable(Double.parseDouble(data));

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

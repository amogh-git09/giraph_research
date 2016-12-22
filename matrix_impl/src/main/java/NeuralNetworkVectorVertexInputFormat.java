import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.VertexInputFormat;
import org.apache.giraph.io.VertexReader;
import org.apache.giraph.io.formats.GiraphTextInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DoubleWritable;
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
    static final int OUTPUT_LAYER = -1;
    static final int INPUT_LAYER = 1;


    @Override
    public VertexReader<Text, NeuronValue, NullWritable> createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return null;
    }

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public List<InputSplit> getSplits(JobContext context, int minSplitCountHint) throws IOException, InterruptedException {
        return null;
    }

    public class NeuralNetworkVertexReader extends VertexReader<Text, NeuronValue, DoubleWritable> {
        private TaskAttemptContext context;
        private RecordReader<LongWritable, Text> lineRecordReader;

        @Override
        public void initialize(InputSplit inputSplit, TaskAttemptContext context) throws IOException, InterruptedException {
            this.context = context;
            lineRecordReader = new LineRecordReader();
            lineRecordReader.initialize(inputSplit, context);
        }

        @Override
        public boolean nextVertex() throws IOException, InterruptedException {
            
        }

        @Override
        public Vertex<Text, NeuronValue, DoubleWritable> getCurrentVertex() throws IOException, InterruptedException {
            return null;
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

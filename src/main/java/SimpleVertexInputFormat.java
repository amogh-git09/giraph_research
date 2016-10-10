import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.Vertex;
import org.apache.giraph.io.VertexInputFormat;
import org.apache.giraph.io.VertexReader;
import org.apache.giraph.io.formats.GiraphTextInputFormat;
import org.apache.hadoop.conf.Configuration;
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
public class SimpleVertexInputFormat extends VertexInputFormat<Text, IntWritable, IntWritable> {
    GiraphTextInputFormat textInputFormat = new GiraphTextInputFormat();

    @Override
    public VertexReader<Text, IntWritable, IntWritable> createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return new SimpleVertexReader();
    }

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public List<InputSplit> getSplits(JobContext context, int minSplitCountHint) throws IOException, InterruptedException {
        return textInputFormat.getVertexSplits(context);
    }

    public class SimpleVertexReader extends VertexReader<Text, IntWritable, IntWritable> {
        private RecordReader<LongWritable, Text> lineRecordReader;
        private TaskAttemptContext context;
        private int networkNum = 1;

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
        public Vertex<Text, IntWritable, IntWritable> getCurrentVertex() throws IOException, InterruptedException {
            Text line = lineRecordReader.getCurrentValue();
            Vertex<Text, IntWritable, IntWritable> vertex = getConf().createVertex();
            String[] words = line.toString().split(" ");
            Text id = new Text(words[0]);
            IntWritable age = new IntWritable(Integer.parseInt(words[1]));

            List<Edge<Text, IntWritable>> edges = new ArrayList<Edge<Text, IntWritable>>();
            for(int i=0; i<words.length-1; i = i+2) {
                Text destId = new Text(words[i]);
                IntWritable numMentions = new IntWritable(Integer.parseInt(words[i+1]));
                edges.add(EdgeFactory.create(destId, numMentions));
            }
            vertex.initialize(id, age, edges);
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

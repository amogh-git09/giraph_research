import org.apache.giraph.edge.Edge;
import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.io.formats.TextVertexInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by amogh-lab on 16/10/09.
 */
public class SimpleTextVertexInputFormat extends
        TextVertexInputFormat<Text, IntWritable, IntWritable> {

    @Override
    public void checkInputSpecs(Configuration conf) {

    }

    @Override
    public TextVertexReader createVertexReader(InputSplit split, TaskAttemptContext context) throws IOException {
        return new SimpleTextVertexReader();
    }

    public class SimpleTextVertexReader extends
            TextVertexReaderFromEachLineProcessed<String[]> {

        private Text id;
        private IntWritable age;
        private List<Edge<Text, IntWritable>> edges;

        @Override
        protected String[] preprocessLine(Text line) throws IOException {
            String[] words = line.toString().split(" ");
            id = new Text(words[0]);
            age = new IntWritable(Integer.parseInt(words[1]));
            edges = new ArrayList<>();
            for(int n = 2; n < words.length-1; n=n+2) {
                Text destId = new Text(words[n]);
                IntWritable numMentions = new IntWritable(Integer.parseInt(words[n+1]));
                edges.add(EdgeFactory.create(destId, numMentions));
            }

            return words;
        }

        @Override
        protected Iterable<Edge<Text, IntWritable>> getEdges(String[] tokens) throws IOException {
            return edges;
        }

        @Override
        protected IntWritable getValue(String[] tokens) throws IOException {
            return age;
        }

        @Override
        protected Text getId(String[] tokens) throws IOException {
            return id;
        }
    }
}

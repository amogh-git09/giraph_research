import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/09/17.
 */
public class GenerateTwitterParallel extends
        BasicComputation<Text, DoubleWritable, DoubleWritable, Text>{

    private static final String[] twitterMembers = {"", "John", "Peter",
            "Mark", "Anne", "Natalie", "Jack", "Julia"};

    private static final byte[][] twitterFollowership = {{0}, {2},
            {}, {1, 4}, {2, 7}, {1, 2, 4}, {3, 4}, {3, 5}};

    @Override
    public void compute(Vertex<Text, DoubleWritable, DoubleWritable> vertex, Iterable<Text> messages) throws IOException {
        if (getSuperstep() == 0) {
            for (int i=1; i<twitterMembers.length; i++) {
                Text dstVertexId = new Text(twitterMembers[i]);

                for (byte neighbour : twitterFollowership[i]) {
                    sendMessage(dstVertexId, new Text(twitterMembers[neighbour]));
                }
            }

            removeVertexRequest(new Text("seed"));
        } else {
            for (Text m : messages) {
                vertex.addEdge(EdgeFactory.create(m, new DoubleWritable(0)));
            }
            vertex.voteToHalt();
        }
    }
}

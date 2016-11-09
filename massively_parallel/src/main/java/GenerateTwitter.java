import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/09/17.
 */
public class GenerateTwitter extends
        BasicComputation<Text, DoubleWritable, DoubleWritable, Text>{

    private static final String[] twitterMembers = {"", "John", "Peter",
            "Mark", "Anne", "Natalie", "Jack", "Julia"};
    private static final byte[][] twitterFollowership = {{0}, {2},
            {}, {1, 4}, {2, 7}, {1, 2, 4}, {3, 4}, {3, 5}};

    @Override
    public void compute(Vertex<Text, DoubleWritable,
            DoubleWritable> vertex, Iterable<Text> messages) throws IOException {
        if(getSuperstep() == 0) {
            for (int i=1; i<twitterFollowership.length; i++) {
                Text dstVertexId = new Text(twitterMembers[i]);
                addVertexRequest(dstVertexId, new DoubleWritable(0));

                for (byte neighbour : twitterFollowership[i]) {
                    addEdgeRequest(dstVertexId,
                            EdgeFactory.create(new Text(twitterMembers[neighbour]),
                                    new DoubleWritable(0)));
                }
            }

            removeVertexRequest(new Text("seed"));
        } else {
            System.out.println("This is " + vertex.getId() + ", I am halting.");
            vertex.voteToHalt();
        }
    }
}

import org.apache.giraph.edge.EdgeFactory;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/09/17.
 */
public class Twitter2Facebook extends
        BasicComputation<Text, DoubleWritable, DoubleWritable, Text>{

    static final DoubleWritable ORIGINAL_EDGE = new DoubleWritable(1);
    static final DoubleWritable SYNTHETIC_EDGE = new DoubleWritable(2);

    @Override
    public void compute(Vertex<Text, DoubleWritable,
            DoubleWritable> vertex, Iterable<Text> messages) throws IOException {
        if(getSuperstep() == 0) {
            sendMessageToAllEdges(vertex, vertex.getId());
        } else {
            for (Text m : messages) {
                DoubleWritable edgeValue = vertex.getEdgeValue(m);
                if (edgeValue == null) {
                    vertex.addEdge(EdgeFactory.create(m, SYNTHETIC_EDGE));
                } else {
                    vertex.setEdgeValue(m, ORIGINAL_EDGE);
                }
            }
            vertex.voteToHalt();
        }
    }
}

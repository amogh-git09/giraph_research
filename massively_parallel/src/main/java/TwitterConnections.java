import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

/**
 * Created by amogh-lab on 16/09/17.
 */
public class TwitterConnections extends
        BasicComputation<Text, DoubleWritable, DoubleWritable, Text>{

    @Override
    public void compute(Vertex<Text, DoubleWritable,
            DoubleWritable> vertex, Iterable<Text> messages)  {
        if(getSuperstep() == 0) {
            vertex.setValue(new DoubleWritable(vertex.getNumEdges()));
            sendMessageToAllEdges(vertex, new Text());
        } else {
            int inDegree = 0;
            for (Text m: messages) {
                inDegree++;
            }
            vertex.setValue(new DoubleWritable(vertex.getValue().get() + inDegree));
            System.out.println("Vertex: " + vertex.getId() +
                    ", Connections: " + vertex.getValue());
            vertex.voteToHalt();
        }
    }
}

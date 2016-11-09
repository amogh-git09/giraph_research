package neural_net;

import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;

import java.io.IOException;

public class NeuralNetworkHelloWorld extends
        BasicComputation<Text, DoubleWritable, DoubleWritable, NullWritable> {

    @Override
    public void compute(Vertex<Text, DoubleWritable, DoubleWritable> vertex,
                        Iterable<NullWritable> messages) throws IOException {

        System.out.print("Hello world from: " + vertex.getId());
        System.out.println("\tvalue = " + vertex.getValue());

        vertex.voteToHalt();
    }
}
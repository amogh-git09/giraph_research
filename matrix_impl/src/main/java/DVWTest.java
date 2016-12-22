import no.uib.cipr.matrix.DenseVector;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;

import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by amogh09 on 16/12/19.
 */
public class DVWTest extends
        BasicComputation<IntWritable, IntWritable, NullWritable, DenseVectorWritable>{

    @Override
    public void compute(Vertex<IntWritable, IntWritable, NullWritable> vertex,
                        Iterable<DenseVectorWritable> messages) throws IOException {

        long ss = getSuperstep();

        System.out.println("This is " + vertex.getId());
        if(ss == 0) {
            DenseVector v = new DenseVector(getRandVector(5));
            DenseVectorWritable vec = new DenseVectorWritable(v);
            System.out.println("Generated vector:");
            printMatrix(vec);

            for(Edge<IntWritable, NullWritable> e : vertex.getEdges()) {
                sendMessage(e.getTargetVertexId(), vec);
            }
        } else {
            for(DenseVectorWritable m : messages) {
                System.out.println("Received message: ");
                printMatrix(m);
            }

            vertex.voteToHalt();
        }
    }

    public void printMatrix(DenseVectorWritable vec) {
        double[] array = vec.vector.getData();

        for(int i=0; i<array.length; i++) {
            System.out.print(array[i] + "  ");
        }
        System.out.println("");
    }

    public double[] getRandVector(int length) {
        double[] vector = new double[length];

        for(int i=0; i<length; i++) {
            vector[i] = generateIntInRange(0, 10);
        }

        return vector;
    }

    public int generateIntInRange(int min, int max) {
        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }
}

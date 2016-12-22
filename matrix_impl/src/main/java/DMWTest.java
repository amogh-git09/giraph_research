import no.uib.cipr.matrix.DenseMatrix;
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
public class DMWTest extends
        BasicComputation<IntWritable, IntWritable, NullWritable, DenseMatrixWritable> {

    @Override
    public void compute(Vertex<IntWritable, IntWritable, NullWritable> vertex,
                        Iterable<DenseMatrixWritable> messages) throws IOException {

        long ss = getSuperstep();

        System.out.println("This is " + vertex.getId());
        if(ss == 0) {
            DenseMatrix m = new DenseMatrix(getRandMatrix(5, 5));
            DenseMatrixWritable vec = new DenseMatrixWritable(m);
            System.out.println("Generated matrix:");
            printMatrix(vec);

            for(Edge<IntWritable, NullWritable> e : vertex.getEdges()) {
                sendMessage(e.getTargetVertexId(), vec);
            }
        } else {
            for(DenseMatrixWritable m : messages) {
                System.out.println("Received message: ");
                printMatrix(m);
            }

            vertex.voteToHalt();
        }
    }

    public void printMatrix(DenseMatrixWritable matrix) {
        int rows = matrix.getNumRows();
        int cols = matrix.getNumCols();
        double[] array = matrix.matrix.getData();

        int k=0;
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                System.out.printf("%02.1f  ", array[k++]);
            }
            System.out.println("");
        }
    }

    public double[][] getRandMatrix(int rows, int cols) {
        double[][] matrix = new double[rows][cols];

        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                matrix[i][j] = generateIntInRange(0, 10);
            }
        }

        return matrix;
    }

    public int generateIntInRange(int min, int max) {
        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }
}

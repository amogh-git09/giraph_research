import org.apache.giraph.conf.LongConfOption;
import org.apache.giraph.edge.Edge;
import org.apache.giraph.graph.BasicComputation;
import org.apache.giraph.graph.Vertex;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;

import java.io.IOException;

/**
 * Created by amogh-lab on 16/09/18.
 */
public class SimpleShortestPathsComputation extends
        BasicComputation<LongWritable, DoubleWritable, FloatWritable, DoubleWritable> {
    public static final LongConfOption SOURCE_ID = new LongConfOption(
            "SimplestShortestPathsVertex.sourceId", 1, "The shortest paths id");

    private static final Logger LOG = Logger.getLogger(SimpleShortestPathsComputation.class);

    private boolean isSource(Vertex<LongWritable, ?, ?> vertex) {
        return vertex.getId().get() == SOURCE_ID.get(getConf());
    }

    @Override
    public void compute(Vertex<LongWritable, DoubleWritable, FloatWritable> vertex,
                        Iterable<DoubleWritable> messages) throws IOException {
        if (getSuperstep() == 0) {
            vertex.setValue(new DoubleWritable(Double.MAX_VALUE));
        }

        Double minDist = isSource(vertex) ? 0d : Double.MAX_VALUE;
        for (DoubleWritable m : messages) {
            minDist = Math.min(minDist, m.get());
        }

        System.out.println("Vertex " + vertex.getId() + " got minDist = " +
                minDist + " vertex value = " + vertex.getValue());

        if (minDist < vertex.getValue().get()) {
            vertex.setValue(new DoubleWritable(minDist));
            System.out.printf("Vertex %d sending messages\n", vertex.getId().get());
            for (Edge<LongWritable, FloatWritable> e : vertex.getEdges()) {
                double distance = minDist + e.getValue().get();
                System.out.println("Vertex " + vertex.getId() + " sent to "
                        + e.getTargetVertexId() + " = " + distance);

                sendMessage(e.getTargetVertexId(), new DoubleWritable(distance));
            }
        } else {
            System.out.printf("Vertex %d is here in else\n", vertex.getId().get());
        }
        vertex.voteToHalt();
    }
}

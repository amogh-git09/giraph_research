import junit.framework.Assert;
import org.apache.giraph.conf.GiraphConfiguration;
import org.apache.giraph.io.formats.AdjacencyListTextVertexOutputFormat;
import org.apache.giraph.io.formats.TextDoubleDoubleAdjacencyListVertexInputFormat;
import org.apache.giraph.utils.InternalVertexRunner;
import org.junit.Test;

/**
 * Created by amogh-lab on 16/09/17.
 */
public class TestGiraphApp {
    final static String[] graphSeed = new String[] {"seed\t0"};
    final static int EXPECTED_VERTICES = 7;

    @Test
    public void testNumberOfVertices() throws Exception {
        GiraphConfiguration conf = new GiraphConfiguration();
        conf.setComputationClass(GenerateTwitterParallel.class);
        conf.setVertexInputFormatClass(TextDoubleDoubleAdjacencyListVertexInputFormat.class);
        conf.setVertexOutputFormatClass(AdjacencyListTextVertexOutputFormat.class);

        Iterable<String> results = InternalVertexRunner.run(conf, graphSeed);

        int totalVertices = 0;
        for (String s : results) {
            System.out.println(s);
            totalVertices++;
        }

        Assert.assertEquals(EXPECTED_VERTICES, totalVertices);
    }
}

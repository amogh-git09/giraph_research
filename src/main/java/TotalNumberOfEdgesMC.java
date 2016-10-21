import neural_net.PersistentIntDenseMatrixSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.IntDenseMatrix;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVectorSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;

/**
 * Created by amogh-lab on 16/09/18.
 */
public class TotalNumberOfEdgesMC extends DefaultMasterCompute {
    public static final String ID = "TotalNumberOfEdgesAggregator";
    public static final String ID2 = "MatrixIntAggregator";
    public static PersistentIntDenseMatrixSumAggregator agg = new PersistentIntDenseMatrixSumAggregator("matrix");

    @Override
    public void compute() {
        System.out.println("SS = " + getSuperstep());
        IntDenseMatrix matrix = agg.getMatrix(2, this);
        System.out.printf("rows = %d, cols = %d\n", matrix.getNumRows(), matrix.getNumColumns());
        for(int i=0; i<matrix.getNumRows(); i++) {
            for(int j=0; j<3; j++) {
                System.out.printf("%2d  ", matrix.get(i, j));
            }
            System.out.println("");
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerAggregator(ID, MyLongSumAggregator.class);
        registerPersistentAggregator(ID2, IntDenseVectorSumAggregator.class);
        agg.registerPersistent(5, this);
    }
}

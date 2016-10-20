import org.apache.giraph.aggregators.matrix.dense.IntDenseMatrix;
import org.apache.giraph.aggregators.matrix.dense.IntDenseMatrixSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVector;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVectorSumAggregator;
import org.apache.giraph.master.DefaultMasterCompute;

/**
 * Created by amogh-lab on 16/09/18.
 */
public class TotalNumberOfEdgesMC extends DefaultMasterCompute {
    public static final String ID = "TotalNumberOfEdgesAggregator";
    public static final String ID2 = "MatrixIntAggregator";
    public static IntDenseMatrixSumAggregator agg = new IntDenseMatrixSumAggregator("matrix");

    @Override
    public void compute() {
//        System.out.println("Total number of edges at superstep " + getSuperstep() + " is "
//            + getAggregatedValue(ID));
//        IntDenseVector v = getAggregatedValue(TotalNumberOfEdgesMC.ID2);
//
//        for(int i=0; i<5; i++) {
//            System.out.print(v.get(i) + "  ");
//        }
//        System.out.println("");

        IntDenseMatrix matrix =  agg.getMatrix(5, this);
        System.out.printf("rows = %d, cols = %d\n", matrix.getNumRows(), matrix.getNumColumns());
        for(int i=0; i<matrix.getNumRows(); i++) {
            for(int j=0; j<matrix.getNumColumns(); j++) {
                System.out.print(matrix.get(i, j) + "  ");
            }
            System.out.println("");
        }
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerAggregator(ID, MyLongSumAggregator.class);
        registerPersistentAggregator(ID2, IntDenseVectorSumAggregator.class);
        agg.register(5, this);
    }
}

package neural_net;

import org.apache.giraph.aggregators.matrix.dense.IntDenseMatrixSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.IntDenseVectorSumAggregator;
import org.apache.giraph.master.MasterAggregatorUsage;

/**
 * Created by amogh-lab on 16/10/21.
 */
public class PersistentIntDenseMatrixSumAggregator extends IntDenseMatrixSumAggregator{
    /**
     * Create a new matrix aggregator with the given prefix name for the vector
     * aggregators.
     *
     * @param name the prefix for the row vector aggregators
     */
    public PersistentIntDenseMatrixSumAggregator(String name) {
        super(name);
    }

    public void registerPersistent(int numRows, MasterAggregatorUsage master)
            throws InstantiationException, IllegalAccessException {
        for (int i = 0; i < numRows; ++i) {
            boolean success = master.registerPersistentAggregator(getRowAggregatorName(i),
                    IntDenseVectorSumAggregator.class);
            if (!success) {
                throw new RuntimeException("Aggregator already registered");
            }
        }
    }
}

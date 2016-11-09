package neural_net;

import org.apache.giraph.aggregators.matrix.dense.DoubleDenseMatrixSumAggregator;
import org.apache.giraph.aggregators.matrix.dense.DoubleDenseVectorOverrideAggregator;
import org.apache.giraph.master.MasterAggregatorUsage;

/**
 * Created by amogh-lab on 16/10/21.
 */
public class PersistentDoubleDenseMatrixSumAggregator extends DoubleDenseMatrixSumAggregator {
    /**
     * Create a new matrix aggregator with the given prefix name for the vector
     * aggregators.
     *
     * @param name the prefix for the row vector aggregators
     */
    public PersistentDoubleDenseMatrixSumAggregator(String name) {
        super(name);
    }

    /**
     * Register the double vector aggregators, one for each row of the matrix.
     *
     * @param numRows the number of rows
     * @param master the master to register the aggregators
     */
    public void registerPersistent(int numRows, MasterAggregatorUsage master)
            throws InstantiationException, IllegalAccessException {
        for (int i = 0; i < numRows; ++i) {
            boolean success = master.registerPersistentAggregator(getRowAggregatorName(i),
                    DoubleDenseVectorOverrideAggregator.class);
            if (!success) {
                throw new RuntimeException("Aggregator already registered");
            }
        }
    }
}

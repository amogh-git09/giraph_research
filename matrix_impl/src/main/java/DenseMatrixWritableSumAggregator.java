import org.apache.giraph.aggregators.BasicAggregator;

/**
 * Created by amogh09 on 16/12/25.
 */
public class DenseMatrixWritableSumAggregator extends BasicAggregator<DenseMatrixWritable>{

    @Override
    public void aggregate(DenseMatrixWritable value) {
        Logger.d("DenseMatrixWritableSumAggregator aggregate");
        Logger.d(String.format("Self size = %d, other size = %d", getAggregatedValue().getNumRows(),
                value.getNumRows()));
        Backpropagation.printMatrix(getAggregatedValue().getMatrix());

        try {
            getAggregatedValue().add(value);
            Logger.d("aggregation succeeded");
        } catch (IndexOutOfBoundsException e) {
            Logger.d("Caught IndexOutOfBoundsException when aggregating weight matrix. Skipping");
        }
    }

    @Override
    public DenseMatrixWritable createInitialValue() {
        return new DenseMatrixWritable();
    }
}

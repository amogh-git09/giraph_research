import no.uib.cipr.matrix.DenseMatrix;
import org.apache.giraph.aggregators.BasicAggregator;

/**
 * Created by amogh09 on 16/12/25.
 */
public class DenseMatrixWritableSumAggregator extends BasicAggregator<DenseMatrixWritable>{

    @Override
    public void aggregate(DenseMatrixWritable value) {
//        Logger.d("DenseMatrixWritableSumAggregator aggregate");
//        Logger.d(String.format("Self size = %d, other size = %d", getAggregatedValue().getNumRows(),
//                value.getNumRows()));
//        Backpropagation.printMatrix(getAggregatedValue().getMatrix());

        int selfSize  = getAggregatedValue().getNumRows();
        int otherSize = value.getNumRows();

        if(selfSize == 0 || otherSize == 0) {
            if(!(selfSize == 0 && otherSize == 0)) {
                DenseMatrix preserve = selfSize == 0 ? value.getMatrix() : getAggregatedValue().getMatrix();
                getAggregatedValue().setMatrix(preserve);
            }
        } else if(selfSize != otherSize) {
            throw new IllegalArgumentException(String.format("selfSize != otherSize, % != %d",
                    selfSize, otherSize));
        } else {
            getAggregatedValue().add(value);
        }
    }

    @Override
    public DenseMatrixWritable createInitialValue() {
        return new DenseMatrixWritable();
    }
}

import org.apache.giraph.aggregators.BasicAggregator;
import org.apache.hadoop.io.LongWritable;

/**
 * Created by amogh-lab on 16/09/18.
 */
public class MyLongSumAggregator extends BasicAggregator<LongWritable> {
    @Override
    public void aggregate(LongWritable value) {
        getAggregatedValue().set(getAggregatedValue().get() + value.get());
    }

    @Override
    public LongWritable createInitialValue() {
        return new LongWritable(0);
    }
}

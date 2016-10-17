package neural_net;

import org.apache.giraph.aggregators.Aggregator;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;

import java.util.HashMap;

/**
 * Created by amogh-lab on 16/10/17.
 */
public class StringDoubleHashMapAggregator implements Aggregator<Text> {
    public static final String DELIMITER = ":";
    private HashMap<Text, DoubleWritable> hashMap;

    public StringDoubleHashMapAggregator() {
        hashMap = new HashMap<Text, DoubleWritable>();
    }

    @Override
    public void aggregate(Text value) {
        String text = value.toString();
        String[] keyVal = text.split(DELIMITER);
        Text key = new Text(keyVal[0]);
        DoubleWritable val = new DoubleWritable(Double.parseDouble(keyVal[1]));

        hashMap.put(key, val);
    }

    @Override
    public Text createInitialValue() {
        return null;
    }

    public DoubleWritable getAggregatedValue() {

    }

    @Override
    public Text getAggregatedValue() {
        return null;
    }

    @Override
    public void setAggregatedValue(Text value) {

    }

    @Override
    public void reset() {

    }
}

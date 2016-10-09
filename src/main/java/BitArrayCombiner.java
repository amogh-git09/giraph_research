import org.apache.giraph.combiner.MessageCombiner;
import org.apache.hadoop.io.IntWritable;

/**
 * Created by amogh-lab on 16/09/17.
 */
public class BitArrayCombiner extends
        MessageCombiner<IntWritable, IntWritable>{
    @Override
    public void combine(IntWritable vertexIndex,
                        IntWritable originalMessage, IntWritable messageToCombine) {
        originalMessage.set(1<<messageToCombine.get() | originalMessage.get());
    }

    @Override
    public IntWritable createInitialMessage() {
        return null;
    }
}

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh09 on 16/12/19.
 */
public class NeuronValue implements WritableComparable {
    DenseVectorWritable activations;
    DenseMatrixWritable weights;

    public NeuronValue() {}

    public NeuronValue(DenseVectorWritable activations) {
        this.activations = activations;
    }

    public NeuronValue(DenseVectorWritable activations, DenseMatrixWritable weights) {
        this.activations = activations;
        this.weights = weights;
    }

    @Override
    public int compareTo(Object o) {
        return 0;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        activations.write(dataOutput);
        weights.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        activations.readFields(dataInput);
        weights.readFields(dataInput);
    }
}

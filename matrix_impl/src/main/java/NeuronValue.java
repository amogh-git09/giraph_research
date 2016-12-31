import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh09 on 16/12/19.
 */
public class NeuronValue implements WritableComparable {
    private DenseVectorWritable activations;
    private DenseVectorWritable output;

    public NeuronValue() {
        this.activations = new DenseVectorWritable();
        this.output = new DenseVectorWritable();
    }

    public NeuronValue(DenseVectorWritable activations, DenseVectorWritable output) {

        this.activations = activations == null ? new DenseVectorWritable() : activations;
        this.output = output == null ? new DenseVectorWritable() : output;
    }

    @Override
    public int compareTo(Object o) {
        return 0;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        activations.write(dataOutput);
        output.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        activations.readFields(dataInput);
        output.readFields(dataInput);
    }

    public DenseVectorWritable getActivations() {
        return activations;
    }

    public void setActivations(DenseVectorWritable activations) {
        this.activations = activations;
    }

    public DenseVectorWritable getOutput() {
        return output;
    }
}

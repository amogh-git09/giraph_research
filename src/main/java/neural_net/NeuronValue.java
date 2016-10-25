package neural_net;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh-lab on 16/10/25.
 */
public class NeuronValue implements WritableComparable {
    private DoubleWritable activation = new DoubleWritable(0.0D);
    private DoubleWritable error = new DoubleWritable(0.0D);

    public NeuronValue() {

    }

    public NeuronValue(Double a, Double e) {
        activation = new DoubleWritable(a);
        error = new DoubleWritable(e);
    }

    public double getActivation() {
        return activation.get();
    }

    public double getError() {
        return error.get();
    }

    public void setActivation(Double a) {
        activation.set(a);
    }

    public void setError(Double e) {
        error.set(e);
    }

    @Override
    public int compareTo(Object o) {
        NeuronValue other = (NeuronValue) o;
        return activation.compareTo(other.getActivation());
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        activation.write(dataOutput);
        error.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        activation.readFields(dataInput);
        error.readFields(dataInput);
    }
}

package distributed_net;

import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh09 on 16/12/14.
 */
public class EdgeValue implements Writable {
    private DoubleWritable weight = new DoubleWritable();
    private DoubleWritable delta = new DoubleWritable();

    public EdgeValue() {

    }

    public EdgeValue(double weight, double delta) {
        this.weight = new DoubleWritable(weight);
        this.delta = new DoubleWritable(delta);
    }

    public void setWeight(double weight) {
        this.weight.set(weight);
    }

    public void addDelta(double delta) {
        this.delta.set(this.delta.get() + delta);
    }

    public void setDelta(double delta) {
        this.delta.set(delta);
    }

    public double getWeight() {
        return this.weight.get();
    }

    public double getDelta() {
        return this.delta.get();
    }

    public void resetDelta() {
        setDelta(0);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        weight.write(dataOutput);
        delta.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        weight.readFields(dataInput);
        delta.readFields(dataInput);
    }
}

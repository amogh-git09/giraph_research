package distributed_net;

import debug.Logger;
import org.apache.giraph.utils.ArrayWritable;
import org.apache.hadoop.io.*;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh-lab on 16/10/25.
 */
public class NeuronValue implements WritableComparable {
    private DoubleWritable activation = new DoubleWritable(0.0D);
    private DoubleWritable weightedInput = new DoubleWritable(0.0D);
    private DoubleWritable error = new DoubleWritable(0.0D);
    private IntWritable classFlag = new IntWritable(0);
    private ArrayWritable<DoubleWritable> derivatives = new ArrayWritable<>(DoubleWritable.class, new DoubleWritable[0]);

    public NeuronValue() {

    }

    public NeuronValue(Double a, Double w, Double e, int f, int nextLayerNeuronCount) {
        activation = new DoubleWritable(a);
        weightedInput = new DoubleWritable(w);
        error = new DoubleWritable(e);
        classFlag = new IntWritable(f);
        DoubleWritable[] tmp = new DoubleWritable[nextLayerNeuronCount];

        for(int i=0; i<nextLayerNeuronCount; i++) {
            tmp[i] = new DoubleWritable(0);
        }

        this.derivatives.set(tmp);
    }

    public double getActivation() {
        return activation.get();
    }

    public double getWeightedInput() {
        return weightedInput.get();
    }

    public double getError() {
        return error.get();
    }

    public int getClassFlag() {
        return classFlag.get();
    }

    public void setClassFlag(int f) {
        classFlag.set(f);
    }

    public void setActivation(Double a) {
        activation.set(a);
    }

    public void setWeightedInput(Double w) {
        weightedInput.set(w);
    }

    public void setError(Double e) {
        error.set(e);
    }

    public void updateDerivative(int index, double val) {
        double old = derivatives.get()[index].get();
        derivatives.get()[index].set(old + val);
    }

    public int getDerivativesLength() {
        return derivatives.get().length;
    }

    public void flushDerivatives() {
        for(int i=0; i<derivatives.get().length; i++) {
            derivatives.get()[i].set(0);
        }
    }

    public void printDerivatives() {
        for(int i=0; i<derivatives.get().length; i++) {
            Logger.d(String.format("derivatives[%d] = %.15f",
                    i, derivatives.get()[i].get()));
        }
    }

    public double getDerivative(int index) {
        return derivatives.get()[index].get();
    }

    @Override
    public int compareTo(Object o) {
        NeuronValue other = (NeuronValue) o;
        return activation.compareTo(other.getActivation());
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        activation.write(dataOutput);
        weightedInput.write(dataOutput);
        error.write(dataOutput);
        classFlag.write(dataOutput);
        derivatives.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        activation.readFields(dataInput);
        weightedInput.readFields(dataInput);
        error.readFields(dataInput);
        classFlag.readFields(dataInput);
        derivatives.readFields(dataInput);
    }
}

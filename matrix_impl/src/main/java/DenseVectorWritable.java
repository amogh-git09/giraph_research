import no.uib.cipr.matrix.DenseVector;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class DenseVectorWritable implements Writable {
    DenseVector vector;

    public DenseVectorWritable() {
        this.vector = new DenseVector(0);
    }

    public DenseVectorWritable(DenseVector v) {
        this.vector = v;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        double[] data = vector.getData();
        DoubleWritable[] dataWr = new DoubleWritable[data.length];

        for (int i = 0; i < data.length; i++) {
            dataWr[i] = new DoubleWritable(data[i]);
        }

        ArrayWritable array = new ArrayWritable(DoubleWritable.class, dataWr);
        array.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        ArrayWritable array = new ArrayWritable(DoubleWritable.class);
        array.readFields(dataInput);

        DoubleWritable[] dataWr = (DoubleWritable[]) array.toArray();
        double[] data = new double[dataWr.length];

        for (int i = 0; i < data.length; i++) {
            data[i] = dataWr[i].get();
        }

        vector = new DenseVector(data);
    }

    public void printDoubles(double[] data) {
        for (int i = 0; i < data.length; i++) {
            System.out.print(data[i]);
        }

        System.out.println("");
    }
}
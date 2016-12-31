import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import no.uib.cipr.matrix.Matrix;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh09 on 16/12/19.
 */
public class DenseMatrixWritable implements Writable {
    private DenseMatrix matrix;
    private IntWritable numRows, numCols;

    public DenseMatrixWritable() {
        this.matrix = new DenseMatrix(0, 0);
        this.numRows = new IntWritable(0);
        this.numCols = new IntWritable(0);
    }

    public DenseMatrixWritable(DenseMatrix matrix) {
        this.matrix = matrix;
        this.numRows = new IntWritable(matrix.numRows());
        this.numCols = new IntWritable(matrix.numColumns());
    }

    public int getNumRows() {
        return numRows.get();
    }

    public int getNumCols() {
        return numCols.get();
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        numRows.write(dataOutput);
        numCols.write(dataOutput);

        double[] data = matrix.getData();
        DenseVector v = new DenseVector(data);
        DenseVectorWritable vector = new DenseVectorWritable(v);
        vector.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        numRows = new IntWritable();
        numCols = new IntWritable();
        numRows.readFields(dataInput);
        numCols.readFields(dataInput);
        DenseVectorWritable vector = new DenseVectorWritable();
        vector.readFields(dataInput);
        double[] data = vector.vector.getData();
        double[][] structuredData = new double[numRows.get()][numCols.get()];

        int k = 0;
        for (int i = 0; i < numCols.get(); i++) {
            for (int j = 0; j < numRows.get(); j++) {
                structuredData[j][i] = data[k++];
            }
        }

        this.matrix = numCols.get() == 0 ? new DenseMatrix(0, 0) : new DenseMatrix(structuredData);
    }

    public void setMatrix(DenseMatrix matrix) {
        this.matrix = matrix;
        this.numRows = new IntWritable(matrix.numRows());
        this.numCols = new IntWritable(matrix.numColumns());
    }

    public DenseMatrix getMatrix() {
        return matrix;
    }

    public void add(DenseMatrixWritable other) {
        this.matrix.add((Matrix) other.getMatrix());
    }
}

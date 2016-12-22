import no.uib.cipr.matrix.DenseMatrix;
import no.uib.cipr.matrix.DenseVector;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Created by amogh09 on 16/12/19.
 */
public class DenseMatrixWritable implements Writable{
    DenseMatrix matrix;
    private IntWritable numRows, numCols;

    public DenseMatrixWritable() {
        IntWritable numRows = new IntWritable();
        IntWritable numCols = new IntWritable();
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
        for(int i=0; i<numRows.get(); i++) {
            for(int j=0; j<numCols.get(); j++) {
                structuredData[i][j] = data[k++];
            }
        }

        this.matrix = new DenseMatrix(structuredData);
    }
}

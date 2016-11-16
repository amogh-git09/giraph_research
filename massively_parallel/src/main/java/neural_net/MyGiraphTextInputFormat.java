package neural_net;

import org.apache.giraph.io.formats.GiraphTextInputFormat;
import org.apache.hadoop.fs.BlockLocation;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.JobContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by amogh-lab on 16/11/06.
 */
public class MyGiraphTextInputFormat extends GiraphTextInputFormat {

    private static final double SPLIT_SLOP = 1.1; // 10% slop

    private List<InputSplit> myGetSplits(JobContext job, List<FileStatus> files)
            throws IOException {
        long minSize = Math.max(getFormatMinSplitSize(), getMinSplitSize(job));
        long maxSize = getMaxSplitSize(job);

        // generate splits
        List<InputSplit> splits = new ArrayList<InputSplit>();

        for (FileStatus file: files) {
            Path path = file.getPath();
            System.out.println("\n\n\n\npath: " + path);
            FileSystem fs = path.getFileSystem(job.getConfiguration());
            long length = file.getLen();
            BlockLocation[] blkLocations = fs.getFileBlockLocations(file, 0, length);
            if ((length != 0) && isSplitable(job, path)) {
                System.out.println("Splittable!");
                long blockSize = file.getBlockSize();
                long splitSize = computeSplitSize(blockSize, minSize, maxSize);

                long bytesRemaining = length;

                System.out.printf("blockSize = %d, splitSize = %d, bytesRemaining = %d\n", blockSize, splitSize, bytesRemaining);

                while (((double) bytesRemaining) / splitSize > SPLIT_SLOP) {
                    int blkIndex = getBlockIndex(blkLocations, length - bytesRemaining);
                    splits.add(new FileSplit(path, length - bytesRemaining, splitSize,
                            blkLocations[blkIndex].getHosts()));
                    bytesRemaining -= splitSize;
                }

                if (bytesRemaining != 0) {
                    splits.add(new FileSplit(path, length - bytesRemaining,
                            bytesRemaining,
                            blkLocations[blkLocations.length - 1].getHosts()));
                }
            } else if (length != 0) {
                splits.add(new FileSplit(path, 0, length, blkLocations[0].getHosts()));
            } else {
                //Create empty hosts array for zero length files
                splits.add(new FileSplit(path, 0, length, new String[0]));
            }
        }
        return splits;
    }

    @Override
    public List<InputSplit> getVertexSplits(JobContext job) throws IOException {
        List<FileStatus> files = listVertexStatus(job);
        List<InputSplit> splits = myGetSplits(job, files);
        // Save the number of input files in the job-conf
        job.getConfiguration().setLong(NUM_VERTEX_INPUT_FILES, files.size());
        System.out.println("Total # of vertex splits: " + splits.size());
        return splits;
    }
}

package neural_net;

import org.apache.commons.io.FilenameUtils;

import java.io.*;

/**
 * Created by amogh-lab on 16/10/27.
 */
public class CSVToUsableFormat {
    public static void Convert(String inputFileName) {
        String ext = FilenameUtils.getExtension(inputFileName);
        String name = FilenameUtils.getBaseName(inputFileName);
        String outputFileName = name + "_cnv." + ext;
        File file = new File(outputFileName);
        PrintWriter writer = null;
        try {
            writer = new PrintWriter(file, "UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
        int counter = 0;
        int minClassCount = 0;
        int maxClassCount = 2;
        int expectedTokenCnt = 5;

        //anomaly detection
        try (BufferedReader br = new BufferedReader(new FileReader(inputFileName))) {
            String line;
            while((line = br.readLine()) != null) {
                if(line.equals("")) continue;
                String[] tokens = line.split(",");
                if(tokens.length != expectedTokenCnt) {
                    throw new IllegalArgumentException("Unexpected line: " + line +
                            "\nThere is an anomalous data sample.");
                }

//                int cls = Integer.parseInt(tokens[tokens.length-1]);
//                if(cls == 0) {
//                    System.out.println("class 0 exists");
//                }
//                maxClassCount = Math.max(maxClassCount, cls);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        try {
            try (BufferedReader br = new BufferedReader(new FileReader(inputFileName))) {
                String line;
                while((line = br.readLine()) != null) {
                    if(line.equals("")) continue;
                    counter++;
                    if(counter >= 5000) {
                        System.out.println("breaking");
                        break;
                    }

                    if(counter%100 == 0) {
                        System.out.println("Processed " + counter + " lines");
                    }

                    String[] tokens = line.split(",");
                    int i;
                    for(i=0; i<tokens.length-1; i++) {
                        writer.write(tokens[i] + "\n");
                    }
                    writer.write("output\n");
                    for(int j=minClassCount; j<=maxClassCount; j++) {
                        int cls = -1;
                        switch (tokens[i]) {
                            case "Iris-setosa":
                                cls = 0;
                                break;
                            case "Iris-versicolor":
                                cls = 1;
                                break;
                            case "Iris-virginica":
                                cls = 2;
                                break;
                        }
                        if(cls == j)
                            writer.write(1 + "\n");
                        else
                            writer.write(0 + "\n");
                    }
                    writer.write("done\n");
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            writer.close();
        }

        System.out.println("classes = " + maxClassCount);
    }

    public static void main(String[] args) throws IOException {
        if(args.length == 0) {
            System.out.println("You did not specify any input files to convert.");
            System.exit(0);
        }

        Convert(args[0]);
    }
}

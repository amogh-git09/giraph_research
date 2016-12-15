package database_input_helper;

import config.Config;
import distributed_net.DistNeuralNet;
import org.apache.commons.lang.StringUtils;
import redis.clients.jedis.Jedis;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class RedisInputHelper {
    final static int featureCount = Config.INPUT_LAYER_NEURON_COUNT;
    final static int dataSetSize = 1;

    public static void main(String[] args) throws IOException {
        Jedis jedis = new Jedis("localhost");
        System.out.println("Connection to server successful");
        System.out.println("Server is runnning: " + jedis.ping());
        jedis.flushAll();
//        inputTestData(jedis, 1, 2);
        inputIrisData(jedis, "iris.txt");

    }

    private static void inputIrisData(Jedis jedis, String file) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            int dataSetIndex = 1;
            while ((line = br.readLine()) != null) {
                if(StringUtils.isEmpty(line))
                    continue;

                // process the line.
                String[] tokens = line.split(",");
                int i = 1;
                for(; i<tokens.length; i++) {
                    String key = String.format("%d:i:%d", dataSetIndex, i);
                    String val = Double.parseDouble(tokens[i-1]) + "";
                    jedis.set(key, val);
                    System.out.printf("Key -> %s,  val -> %s\n", key, val);
                }

                String key = String.format("%d:o", dataSetIndex);
                String val = "";
                switch (tokens[i-1]) {
                    case "Iris-setosa":
                        val = "1";
                        break;
                    case "Iris-versicolor":
                        val = "2";
                        break;
                    case "Iris-virginica":
                        val = "3";
                        break;
                }

                System.out.printf("Key -> %s,  val -> %s\n", key, val);
                jedis.set(key, val);
                dataSetIndex++;
            }

            System.out.println("Set " + dataSetIndex + " values");
        }
    }

    private static void inputTestData(Jedis jedis, int dataSetSize, int featureCount) {
        for(int i=1; i<=dataSetSize; i++) {
            for(int j=1; j<=featureCount; j++) {
                if(j == 1) {
                    String key = i + ":" + "i" + ":" + j;
                    jedis.set(key, 1 + "");
                } else if(j == 2) {
                    String key = i + ":" + "i" + ":" + j;
                    jedis.set(key, 5 + "");
                }
            }

            String key = i + ":" + "o";
            jedis.set(key, 0 + "");
        }
    }
}

package database_input_helper;

import redis.clients.jedis.Jedis;

public class RedisInputHelper {
    final static int featureCount = 10;
    final static int dataSetSize = 1;

    public static void main(String[] args) {
        Jedis jedis = new Jedis("localhost");
        System.out.println("Connection to server successful");
        System.out.println("Server is runnning: " + jedis.ping());

        for(int i=1; i<=dataSetSize; i++) {
            for(int j=1; j<=featureCount; j++) {
                String key = i + ":" + j;
                jedis.set(key, i + j + "");
            }
        }
    }
}

package worker_context;

import debug.Logger;
import org.apache.commons.beanutils.converters.IntegerConverter;
import org.apache.giraph.worker.WorkerContext;
import redis.clients.jedis.Jedis;

import java.util.Random;

/**
 * Created by amogh-lab on 16/11/09.
 */
public class RedisWorkerContext extends WorkerContext {
    public Jedis jedis;
    public Random random;

    public final double EPSILON = 0.001;
    private static final String REDIS_SERVER = "localhost";


    @Override
    public void preApplication() throws InstantiationException, IllegalAccessException {
        jedis = new Jedis(REDIS_SERVER);
        random = new Random();
    }

    @Override
    public void postApplication() {

    }

    @Override
    public void preSuperstep() {

    }

    @Override
    public void postSuperstep() {

    }

    public double getInputData(int dataNum, int featureNum) {
        String key = dataNum + ":" + "i" + ":" + featureNum;
        Logger.d("Getting input for key: " + key);
        String val = jedis.get(key);
        return Double.parseDouble(val);
    }

    public int getOutputData(int dataNum, int classNum) {
        String key = dataNum + ":o";
        String val = jedis.get(key);
        return Integer.parseInt(val) == classNum ? 1 : 0;
    }

    public double getRandomWeight() {
        return getRandomInRange(-EPSILON, EPSILON);
    }

    private double getRandomInRange(Double min, Double max) {
        Double rand = random.nextDouble();
        return min + (max - min) * rand;
    }

    public double getRandomActivation() {
        return getRandomInRange(-10d, 10d);
    }
}

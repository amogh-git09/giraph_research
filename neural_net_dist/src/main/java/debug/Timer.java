package debug;

/**
 * Created by amogh-lab on 17/01/08.
 */
public class Timer {
    private long start;
    private long end;
    private boolean running = false;

    public void start() {
        if(!running) {
            this.start = System.currentTimeMillis();
            running = true;
        }
    }

    public void stop() {
        this.end = System.currentTimeMillis();
        running = false;
    }

    public void report() {
        if(running) {
            stop();
        }

        Logger.p(String.format("Took %.3f secs", ((end - start) / 1000d)));
    }
}
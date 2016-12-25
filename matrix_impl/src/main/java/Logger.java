/**
 * Created by amogh-lab on 16/11/09.
 */
public class Logger {
    public static boolean DEBUG = true;
    public static boolean INFO = true;
    public static boolean PERFORMANCE = true;

    public static void d(String msg) {
        if(DEBUG) {
            System.out.println("DEBUG: " + msg);
        }
    }

    public static void p(String msg) {
        if(PERFORMANCE) {
            System.out.println("PERF: " + msg);
        }
    }

    public static void i(String msg) {
        if(INFO) {
            System.out.println("INFO: " + msg);
        }
    }
}

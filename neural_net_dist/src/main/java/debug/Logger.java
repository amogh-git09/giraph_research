package debug;

/**
 * Created by amogh-lab on 16/11/09.
 */
public class Logger {
    public static boolean DEBUG = false;
    public static boolean INFO1 = false;
    public static boolean INFO2 = true;

    public static void d(String msg) {
        if(DEBUG) {
            System.out.println("DEBUG: " + msg);
        }
    }

    public static void i1(String msg) {
        if(INFO1) {
            if(DEBUG)
                System.out.println("\nINFO: " + msg);
            else
                System.out.println("INFO: " + msg);
        }
    }

    public static void i2(String msg) {
        if(INFO2)
            System.out.println("INFO: " + msg);
    }
}

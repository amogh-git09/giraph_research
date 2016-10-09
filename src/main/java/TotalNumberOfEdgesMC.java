import org.apache.giraph.master.DefaultMasterCompute;

/**
 * Created by amogh-lab on 16/09/18.
 */
public class TotalNumberOfEdgesMC extends DefaultMasterCompute {
    public static final String ID = "TotalNumberOfEdgesAggregator";

    @Override
    public void compute() {
        System.out.println("Total number of edges at superstep " + getSuperstep() + " is "
            + getAggregatedValue(ID));
    }

    @Override
    public void initialize() throws InstantiationException, IllegalAccessException {
        registerAggregator(ID, MyLongSumAggregator.class);
    }
}

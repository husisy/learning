import java.util.ArrayList;

public class Main01 {
    public static void main(String[] args) {
        ArrayList<String> z1 = new ArrayList<String>();
        z1.add("2");
        z1.add("3");
        z1.add("3");
        String x1 = "";
        x1 += z1.get(0);
        x1 += z1.get(1);
        x1 += z1.get(2);
        System.out.println(x1);
    }
}

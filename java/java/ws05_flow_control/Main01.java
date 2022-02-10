public class Main01{
    public static void main(String[] args) {
        Main01.java_flow_control();
    }

    private static void java_flow_control() {
        System.out.println("\n#java_flow_control");

        System.out.println("##if-else if-else");
        double x = 233.1;
        if (Math.abs(x - 233) <= 1e-3) {
            System.out.println("abs(x-233) <= 1e-3");
        } else if (x > 233 + 1e-3) {
            System.out.println("x > 233+1e-3");
        } else {
            System.out.println("x < 233-1e-3");
        }

        System.out.println("##switch-case-default");
        int x1 = 1;
        switch (x1) {
        case 0:
            System.out.println("x1==0");
        case 1:
        case 2:
            System.out.println("x1==1");
        default:
            System.out.println("default");
        }

        System.out.println("##while");
        int x2 = 3;
        while (x2 < 5) {
            System.out.printf("x2 = %d\n", x2);
            x2++;
        }
        do {
            System.out.printf("x2 = %d\n", x2);
            x2--;
        } while (x2 > 2);

        System.out.println("##for");
        int[] x3 = { 1, 4, 9 };
        for (int i = 0; i < x3.length; i++) {
            System.out.printf("i = %d\n", i);
        }
        for (int x4 : x3) {
            System.out.printf("x = %d\n", x4);
        }
    }
}
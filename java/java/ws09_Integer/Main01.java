public class Main01 {
    public static void main(String[] args) {
        Main01.misc();
    }

    public static void misc() {
        System.out.println("\n#misc");
        Integer n1 = new Integer(233);
        Integer n2 = Integer.valueOf(233);
        Integer n3 = Integer.valueOf("233");
        Integer n4 = 233;
        String s5 = n1.toString();
        int x1 = n1.intValue();
        int x2 = Integer.parseInt("233");
        int x3 = n1;
        Number n5 = Integer.valueOf(233) + 1;
        System.out.println(n5.floatValue());
    }
}
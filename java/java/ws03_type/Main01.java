public class Main01 {
    public static void main(String[] args) {
        System.out.println("\nmian fang fa...hello world!");
        for (String x : args) {
            System.out.println(x);
            if ("-version".equals(x)) {
                System.out.println("no version yet");
            }
        }

        Main01.int_type();
        Main01.double_type();
        Main01.char_type();
        Main01.boolean_type_op();
        Main01.trans_type();
        Main01.basic_operator();
    }

    private static void int_type() {
        System.out.println("\n#int_type");

        byte b = 127; // -128 ~ 127
        short s = 32767; // -32768 ~ 32767
        int i = 2147483647; // -2147483648 ~ 2147483647
        long l = -9_223_372_036_854_775_808L; // -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807
        System.out.println("##byte/short/int/long type");
        System.out.println(b);
        System.out.println(s);
        System.out.println(i);
        System.out.println(l);

        System.out.println("##binary/octal/decimal/Hexadecimal");
        System.out.println(0b1000000000); // 512
        System.out.println(01000); // 512
        System.out.println(512);
        System.out.println(0x200); // 512

        System.out.println("##Integer.toString()");
        System.out.println(Integer.toBinaryString(512));
        System.out.println(Integer.toOctalString(512));
        System.out.println(Integer.toString(512));
        System.out.println(Integer.toHexString(512));
    }

    private static void double_type() {
        System.out.println("\n#double_type");

        double x1 = 2.33;
        float x2 = 2.33f;
        System.out.println(x1);
        System.out.println(x2);

        System.out.println(2.33e0);
        System.out.println(2.33E1);

        System.out.println(233_233.233_233);
    }

    private static void char_type() {
        System.out.println("\n#char_type");
        
        System.out.println('a');
        // System.out.println(char 61);
        System.out.println('\u0061');
    }

    private static void boolean_type_op() {
        System.out.println("\n#boolean_type_op");
        boolean x1 = true, x2 = false;
        System.out.println(x1);
        System.out.println(x2);
        System.out.println(!true);
        System.out.println(false && (1 / 0 > 0));
        System.out.println(true || (1 / 0 > 0));
        System.out.println(true ? -233 : 233);

        int x3 = 9, x4 = 13;
        System.out.println("x3:    " + Integer.toBinaryString(x3));
        System.out.println("x4:    " + Integer.toBinaryString(x4));
        System.out.println("~x3:   " + Integer.toBinaryString(~x3));
        System.out.println("x3&x4: " + Integer.toBinaryString(x3&x4));
        System.out.println("x3|x4: " + Integer.toBinaryString(x3|x4));
        System.out.println("x3^x4: " + Integer.toBinaryString(x3^x4));
        System.out.println("x3<<1: " + Integer.toBinaryString(x3<<1));
        System.out.println("x3>>1: " + Integer.toBinaryString(x3>>1));
    }

    private static void trans_type() {
        System.out.println("\n#trans_type");

        System.out.println((int) (3.2 + 0.5));
        System.out.println((int) 1e20);
        System.out.println((int) -1e20);
    }

    private static void basic_operator() {
        System.out.println("\n#basic_operator");

        System.out.println("##divide /");
        System.out.println(100 / 9);
        System.out.println(100.0 / 9);
        System.out.println(100 / 9.0);

        System.out.println("##percent %");
        System.out.println(3 % 2);
        System.out.println(4 % 2);
        System.out.println(3.0 % 2);
        
        System.out.println(4%3);
        System.out.println(4%-3);
        System.out.println(-4%3);
        System.out.println(-4%-3);
    }

}
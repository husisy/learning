package basic;

import java.nio.charset.StandardCharsets;
/**
* TODO: Chinese String
*/
import java.util.Arrays;
import java.util.Scanner;

public class Main01 {
    public static void main(String[] args) {
        System.out.println("\nmian fang fa...hello world!");
        for (String x : args) {
            System.out.println(x);
            if ("-version".equals(x)) {
                System.out.println("no version yet");
            }
        }

        Main01.java_operator();
        Main01.java_type();
        Main01.java_trans_type();
        Main01.java_boolean_op();
        Main01.java_string();
        Main01.java_array();
        // Main01.java_io();
        Main01.java_flow_control();
        Main01.java_format();
        Main01.java_subclass();
    }

    private static void java_operator() {
        System.out.println("\nJava_operator");

        System.out.println(100 / 9);
        System.out.println(100.0 / 9);
        System.out.println(100 / 9.0);

        System.out.println(3 % 2);
        System.out.println(4 % 2);
        System.out.println(3.0 % 2);
    }

    private static void java_type() {
        System.out.println("\nJava_type");

        byte b = 127; // -128 ~ 127
        short s = 32767; // -32768 ~ 32767
        int i = 2147483647; // -2147483648 ~ 2147483647
        long l = -9_223_372_036_854_775_808L; // -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807

        System.out.println(b);
        System.out.println(s);
        System.out.println(i);
        System.out.println(l);

        System.out.println(0xff0000);// 16711680
        System.out.println(0b1000000000);// 512
        System.out.println(Integer.toHexString(233)); // e9
        System.out.println(Integer.toBinaryString(233)); // 11101001

        final int MY_233 = 233;
        System.out.println(MY_233);
    }

    private static void java_trans_type() {
        System.out.println("\njava_trans_type");

        System.out.println((int) (3.2 + 0.5));
        System.out.println((int) 1e20);
    }

    private static void java_boolean_op() {
        System.out.println("\njava_boolean_op");
        System.out.println(true);
        System.out.println(false);
        System.out.println(!true);
        System.out.println(false && (1 / 0 > 0));
        System.out.println(true || (1 / 0 > 0));
        System.out.println(true ? -233 : 233);
    }

    private static void java_string() {
        System.out.println("\njava_string");

        System.out.println('A');
        System.out.println((int) ('A'));
        // System.out.println("*"); //TODO chinese support
        System.out.println('\u0041');
        System.out.println('\u4e2d');
        // System.out.println((int) ('*')); //TODO chinese support
        String a = null;
        System.out.println(a);
        System.out.println("");
        System.out.println("hello " + "world");

        String s1 = "hello", s2 = "Hello".toLowerCase();
        String s3 = s1;
        System.out.printf("s1==s2: %s; s1.equals(s2): %s \n", String.valueOf(s1 == s2), String.valueOf(s1.equals(s2)));
        System.out.printf("s1==s3: %s; s1.equals(s3): %s \n", String.valueOf(s1 == s3), String.valueOf(s1.equals(s3)));
        System.out.println(s1.contains("ll"));
        System.out.println(s1.indexOf("ll"));
        System.out.println(s1.indexOf("l"));
        System.out.println(s1.lastIndexOf("l"));
        System.out.println(s1.startsWith("he"));
        System.out.println(s1.endsWith("he"));

        System.out.println(" \t\r\nhello\n\r\t".trim());
        System.out.println("hello world".substring(2, 4));
        System.out.println("hElLo WoRlD".toUpperCase().toLowerCase());
        System.out.println("hello world".replace('l', 'L'));
        System.out.println("hello world".replace("ll", "LL"));
        System.out.println(String.join("~", "A,,B;C ,D".split("[,; ]+")));

        System.out.println("valueOf(int): " + String.valueOf(233));
        System.out.println("valueOf(boolean): " + String.valueOf(true));
        System.out.println("valueOf(object): " + String.valueOf(new Object()));
        System.out.println("parseInt(int): " + String.valueOf(Integer.parseInt("233") - 1));
        System.out.println("valueOf(String): " + String.valueOf(Integer.valueOf("233") + 1));
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

        System.out.println("hello".toCharArray());
        char[] cs1 = { '2', '3', '3' };
        System.out.println(new String(cs1));

        final byte[] bs1 = "hello".getBytes(StandardCharsets.UTF_8); // strange, Fail when use "UTF-8"
        System.out.println(new String(bs1, StandardCharsets.UTF_8));
        // System.out.println(new String("hello".getBytes("UTF-8"), "UTF-8"));
        // System.out.println(new String("hello".getBytes("UTF-8"),
        // StandardCharsets.UTF_8));

        StringBuilder sb1 = new StringBuilder(128);
        for (int i = 0; i < 100; i++) {
            sb1.append(String.valueOf(i));
        }
        System.out.println(sb1.toString());
        StringBuilder sb2 = new StringBuilder(128);
        System.out.println(sb2.append("Mr ").append("233").append("!").insert(0, "Hello, ").toString());
    }

    private static void java_array() {
        System.out.println(("\njava_array"));

        int[] z0 = new int[5];
        int[] z1 = { 2, 0, 3, 0, 0 };
        int[] z2 = z1; // z1, z2 share memory
        z2[4] = 3;
        System.out.println(z2[0]);
        System.out.println(z2[1]);
        System.out.println(z2[2]);
        System.out.println(z2[3]);
        System.out.println(z2[4]);
        System.out.println(z2.length);
        System.out.println(Arrays.toString(z1));
        System.out.println(Arrays.toString(z2));
        Arrays.sort(z2);
        System.out.println(Arrays.toString(z1));
        System.out.println(Arrays.toString(z2));

        int[][] z3 = { { 1 }, { 2, 3 }, { 4, 5, 6 } };
        System.out.println(Arrays.deepToString(z3));
        for (int[] i : z3) {
            System.out.println(Arrays.toString(i));
        }
    }

    private static void java_io() {
        System.out.println("\njava_io");
        Scanner z1 = new Scanner(System.in);

        System.out.print("please input sth: ");
        String z2 = z1.nextLine();
        System.out.print("233: ");
        System.out.println(z2);

        System.out.print("please input double: ");
        double z3 = z1.nextDouble();
        System.out.printf("233: %7.2f\n", z3);

        System.out.printf("%2$s %1$s\n", "world", "hello");
        z1.close();
    }

    private static void java_flow_control() {
        System.out.println("\njava_flow_control");
        double x = 233.1;
        if (Math.abs(x - 233) <= 1e-3) {
            System.out.println("abs(x-233) <= 1e-3");
        } else if (x > 233 + 1e-3) {
            System.out.println("x > 233+1e-3");
        } else {
            System.out.println("x < 233-1e-3");
        }

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

        int x2 = 3;
        while (x2 < 5) {
            System.out.printf("x2 = %d\n", x2);
            x2++;
        }
        do {
            System.out.printf("x2 = %d\n", x2);
            x2--;
        } while (x2 > 2);

        int[] x3 = { 1, 4, 9 };
        for (int i = 0; i < x3.length; i++) {
            System.out.printf("i = %d\n", i);
        }
        for (int x4 : x3) {
            System.out.printf("x = %d\n", x4);
        }
    }

    private static void java_format() {
        System.out.println("\njava_format");

        System.out.printf("%s is %d years old\n", "Bob", 233);
        System.out.printf("%d; %x\n", 233, 233, 2.33); // 233; e9
        System.out.printf("%f; %7.2f; %e; %+.2f\n", 233.0, 233.0, 233.0, 233.0);
        System.out.printf("%2$d %3$d %1$d\n", 3, 2, 3);
    }

    private static void java_subclass() {
        System.out.println("\nJava subclass");

        Main01 m = new Main01();
        Main01.SubClass x = m.new SubClass();
        x.fn1();
    }

    public class SubClass {
        void fn1() {
            System.out.println("233: ");
        }
    }
}

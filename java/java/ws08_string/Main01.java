import java.nio.charset.StandardCharsets;

public class Main01 {
    public static void main(String[] args) {
        Main01.trans_type();
        Main01.basic_string();
        Main01.char_array();
        Main01.utf8_byte();
        Main01.basic_method();
        Main01.string_builder();
    }

    public static void trans_type() {
        System.out.println("\n#trans_type");

        System.out.println("valueOf(int): " + String.valueOf(233));
        System.out.println("valueOf(boolean): " + String.valueOf(true));
        System.out.println("valueOf(object): " + String.valueOf(new Object()));
        System.out.println("parseInt(int): " + String.valueOf(Integer.parseInt("233") - 1));
        System.out.println("valueOf(String): " + String.valueOf(Integer.valueOf("233") + 1));
    }

    public static void basic_string() {
        System.out.println("\n#basic_string");

        System.out.println('A');
        System.out.println((int) ('A'));
        System.out.println("中");
        System.out.println('\u0041');
        System.out.println('\u4e2d'); // '中'
        // System.out.println((int) ('中'));
        String a = null;
        System.out.println(a);
        System.out.println("");
        System.out.println("hello " + "world");
    }

    public static void char_array() {
        System.out.println("\n#char_array");

        System.out.println("hello".toCharArray());
        char[] cs1 = { '2', '3', '3' };
        System.out.println(new String(cs1));
    }

    public static void utf8_byte() {
        System.out.println("\n#utf8_byte");
        
        final byte[] bs1 = "hello".getBytes(StandardCharsets.UTF_8); // strange, Fail when use "UTF-8"
        System.out.println(new String(bs1, StandardCharsets.UTF_8));
        // System.out.println(new String("hello".getBytes("UTF-8"), "UTF-8"));
        // System.out.println(new String("hello".getBytes("UTF-8"),
        // StandardCharsets.UTF_8));
    }

    public static void basic_method() {
        System.out.println("\n#basic_method");

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
    }

    public static void string_builder() {
        System.out.println("\n#string_builder");

        StringBuilder z1 = new StringBuilder(128);
        for (int i = 0; i < 100; i++) {
            z1.append(String.valueOf(i));
        }
        System.out.println(z1.toString());
        StringBuilder z2 = new StringBuilder(128);
        System.out.println(z2.append("Mr ").append("233").append("!").insert(0, "Hello, ").toString());
    }
}
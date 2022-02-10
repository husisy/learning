package objectOrientedProgramming;

public class Main01 {
    public static void main(String[] args) {
        for (String x1 : args) {
            System.out.println(x1);
        }
        Main01.java_oop_constructor();
        Main01.java_oop_extend();
        Main01.java_oop_abstract();
        Main01.java_oop_interface();
    }

    private static void java_oop_constructor() {
        System.out.println("\njava_oop_constructor");
        ZC1 x1 = new ZC1();
        System.out.println(x1);
        System.err.println(ZC1.getNumber());

        x1.p1 = "olleh";
        System.out.println(x1);

        x1.set_p2("dlorw");
        System.out.println(x1);

        ZC1 x2 = new ZC1("hello", "world");
        System.out.println(x2);
        System.err.println(ZC1.getNumber());
    }

    private static void java_oop_extend() {
        System.out.println("\njava_oop_extend");
        ZC1 x1 = new ZC1("hello", "world");
        ZC2 x2 = new ZC2("hello", "world");
        ZC1 x3;

        System.out.println("ZC1 = ZC1(xx,xx)");
        x3 = x1;
        System.out.println(x3 instanceof ZC1);
        System.out.println(x3 instanceof ZC2);
        System.out.println(x3);

        System.out.println("ZC1 = ZC2(xx,xx)");
        x3 = x2;
        System.out.println(x3 instanceof ZC1);
        System.out.println(x3 instanceof ZC2);
        System.out.println(x3);
    }

    private static void java_oop_abstract() {
        System.out.println("\njava_oop_abstract");

        ZCAbstract00 x1 = new ZCAbstract01();
        System.out.println(x1.hf0());
        System.out.println(x1.hf1());

        x1 = new ZCAbstract02();
        System.out.println(x1.hf0());
        System.out.println(x1.hf1());
    }

    private static void java_oop_interface() {
        System.out.println("\njava_oop_interface");

        ZCInterface00 x1 = new ZCInterface01();
        System.out.println(x1.hf1());

        x1 = new ZCInterface02();
        System.out.println(x1.hf1());
    }
}

class ZC1 {
    public String p1;
    private static int num = 0;

    public ZC1() {
        p1 = "hello";
        p2 = "world";
        num++;
    }

    public ZC1(String s1, String s2) {
        p1 = s1;
        p2 = s2;
        num++;
    }

    public void set_p2(String s) {
        p2 = s;
    }

    public String get_p2() {
        return p2;
    }

    public String toString() {
        return "ZC1(" + p1 + ", " + p2 + ")";
    }

    public static int getNumber() {
        return num;
    }

    private String p2;
}

class ZC2 extends ZC1 {
    public ZC2() {
        super();
    }

    public ZC2(String s1, String s2) {
        super(s1, s2);
    }

    @Override
    public String toString() {
        return "ZC2(" + this.p1 + ", " + this.get_p2() + ")";
    }
}

abstract class ZCAbstract00 {
    public String property = "ZCAbstract00";
    public String hf0(){
        return "hf0() in ZCAbstract00; property: " + this.property;
    }
    public abstract String hf1();
}

class ZCAbstract01 extends ZCAbstract00 {
    @Override
    public String hf0(){
        return "hf0() in ZCAbstract01; property: " + property;
    }

    @Override
    public String hf1() {
        return "hf1() in ZCAbstract01";
    }
}

class ZCAbstract02 extends ZCAbstract00 {
    public ZCAbstract02(){
        property = "ZCAbstract02";
    }

    @Override
    public String hf1() {
        return "hf2() in ZCAbstract02";
    }
}

interface ZCInterface00 {
    String hf1();
    default String hf0(){
        return "233";
    }
}

class ZCInterface01 implements ZCInterface00 {
    public String hf1() {
        return "hf1() in ZCInterface01";
    }
}

class ZCInterface02 implements ZCInterface00 {
    public String hf1() {
        return "hf1() in ZCInterface02";
    }
}
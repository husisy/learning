public class Main01 {
    public static void main(String[] args) {
        Main01.oob_this();
    }

    public static void oob_this() {
        System.out.println("\n#oob_this");

        System.out.println("##coustruct with this");
        ZC1 z1 = new ZC1();
        ZC1 z2 = new ZC1(234, "234");
        System.out.println("(i1,s1)");
        System.out.printf("z1: (%d,%s)\n", z1.i1, z1.s1);
        System.out.printf("z2: (%d,%s)\n", z2.i1, z2.s1);
    }
}

class ZC1 {
    public int i1;
    public String s1;
    ZC1(int i1, String s1){
        this.i1 = i1;
        this.s1 = s1;
    }
    ZC1(){
        this(233, "233");
    }
}
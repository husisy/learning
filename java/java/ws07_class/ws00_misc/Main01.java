public class Main01 {
    public static void main(String[] args) {
        Main01.class_subclass();
    }

    private static void class_subclass() {
        System.out.println("\n#class_subclass");

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
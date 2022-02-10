import java.lang.reflect.Constructor;
import java.util.Arrays;

public class Main01 {
    public static void main(String[] args) throws Exception {
        Main01.basic();
        Main01.equal_or_instanceof();
        Main01.basic_method();
        Main01.class_constructor();
    }

    public static void basic() throws Exception {
        System.out.println("\n#CLASS_basic");
        Class z1 = String.class;
        Class z2 = "233".getClass();
        Class z3 = Class.forName("java.lang.String");

        System.out.println(z1);
        System.out.println(z2);
        System.out.println(z3);
        System.out.println(z1==z2);
        System.out.println(z1==z3);
    }
    
    public static void equal_or_instanceof() {
        System.out.println("\n#equal_or_instanceof");

        Integer z1 = new Integer(233);
        System.out.println(z1 instanceof Integer);
        System.out.println(z1 instanceof Number);
        System.out.println(z1.getClass()==Integer.class);
        // System.out.println(z1.getClass()==Number.class); #compile error
    }

    public static void basic_method() throws Exception {
        System.out.println("\n#basic_method");

        Class z1 = String.class;
        System.out.println(z1.getName());
        System.out.println(z1.getSimpleName());
        System.out.println(z1.getPackage().getName());
        
        System.out.println(Runnable.class.isInterface());
        System.out.println(java.time.Month.class.isEnum());
        System.out.println(String[].class.isArray());
        System.out.println(int.class.isPrimitive());

        System.out.println("##superclass");
        System.out.println(Integer.class.getSuperclass());
        System.out.println(Object.class.getSuperclass());
        System.out.println(Runnable.class.getSuperclass());
        System.out.println(Arrays.toString(Integer.class.getInterfaces()));
        System.out.println(java.util.List.class.isInterface());
        System.out.println(Arrays.toString(java.util.List.class.getInterfaces()));
    }

    public static void class_constructor() throws Exception {
        System.out.println("\n#class_constructor");

        System.out.println((String) String.class.newInstance());
        
        Constructor z1 = Integer.class.getConstructor(int.class);
        System.out.println((Integer) z1.newInstance(233));

        Constructor z2 = Integer.class.getConstructor(String.class);
        System.out.println((Integer) z2.newInstance("233"));

    }
}
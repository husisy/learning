import java.util.Scanner;

public class Main01{
    public static void main(String[] args) {
        Main01.io_scanner();
        Main01.io_format();
    }

    public static void io_scanner() {
        System.out.println("\n#io_scanner");

        Scanner z1 = new Scanner(System.in);

        System.out.print("input whatya: ");
        String z4 = z1.nextLine();
        System.out.printf("the whole line: %s\n", z4);
 
        System.out.print("input an integer: ");
        int z2 = z1.nextInt();
        System.out.printf("square of %d is: %d\n", z2, z2*z2);

        System.out.print("input an double: ");
        double z3 = z1.nextDouble();
        System.out.printf("square of %f is: %f\n", z3, z3*z3);

        z1.close();
    }

    public static void io_format() {
        System.out.println("\nio_format");

        System.out.printf("%s is %d years old\n", "Bob", 233);
        System.out.printf("%d; %x\n", 233, 233, 2.33); // 233; e9
        System.out.printf("%f; %7.2f; %e; %+.2f\n", 233.0, 233.0, 233.0, 233.0);
        System.out.printf("%2$d %3$d %1$d\n", 3, 2, 3);
    }
}
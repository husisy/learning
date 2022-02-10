import java.util.Arrays;

public class Main01{
    public static void main(String[] args) {
        Main01.java_array();
    }

    private static void java_array() {
        System.out.println(("\n#java_array"));

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
}

// import java.io.*;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

class Draft00 {
    // usage:
    // javac draft00.java
    // java draft00
    public static void main(String args[]) {
        System.out.println("Hello, World");

        Draft00.test_writeList();
    }

    public static void test_writeList() {
        try {
            // The FileWriter constructor throws IOException, which must be caught.
            PrintWriter out = new PrintWriter(new FileWriter("tbd00.txt"));
            List<Integer> list = Arrays.asList(2, 3, 3);

            for (int i = 0; i < list.size(); i++) {
                // The get(int) method throws IndexOutOfBoundsException, which must be caught.
                out.println("Value at: " + i + " = " + list.get(i));
            }
            out.close();
        } catch (Exception e) {
        }
    }
}
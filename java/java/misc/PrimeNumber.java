import java.util.Arrays;

public class PrimeNumber{
    public static void main(String[] args) {
        int upperLimit = 10000;
        int[] z1 = new int[upperLimit+1];
        
        //init array to 0,1,2,,...,10000
        for(int i=0; i<z1.length; i++){
            z1[i] = i;
        }

        //set non-prime number to 0
        for(int x=2; x<z1.length; x++){
            for(int y=x*2; y<z1.length; y=y+x){
                z1[y] = 0;
            }
        }

        //use enhanded-for ut print prime number
        for(int x:z1){
            if (x!=0 && x>1){
                System.out.print(x+", ");
            }
        }
    }
}
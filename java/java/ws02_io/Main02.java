import java.awt.*;
import java.awt.event.*;
import javax.swing.*;


public class Main02{
    public static void main(String args[]){
        new AppFrame();
    }
}

class AppFrame extends JFrame{
    JTextField in = new JTextField();
    JButton btn = new JButton("evaluate x square");
    JLabel out = new JLabel("label for showing the result");

    public AppFrame(){
        setLayout(new FlowLayout());
        getContentPane().add(in);
        getContentPane().add(btn);
        getContentPane().add(out);
        btn.addActionListener(new BtnActionAdapter());
        // btn.setSize(200, 50);
        setSize(400,100);
        setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        setVisible(true);
    }

    class BtnActionAdapter implements ActionListener{
        public void actionPerformed(ActionEvent e){
            String s = in.getText();
            double d = Double.parseDouble(s);
            out.setText("square is: " + d*d);
        }
    }
}
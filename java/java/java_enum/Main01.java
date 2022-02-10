
public class Main01{
    public static void main(String[] args){
        for (Weekday day:Weekday.values()){
            System.out.println(day.name());
        }
        Weekday fri = Weekday.FRI;
        System.out.println("FRI.name() = " + fri.name());
        System.out.println("FRI.ordinal() = " + fri.ordinal());
        System.out.println(Weekday.valueOf("FRI").name());
        System.err.println(fri.toChinese());
    }
}

enum Weekday{
    SUN("星期日"), MON("星期一"), TUE("星期二"), WED("星期三"), THU("星期四"), FRI("星期五"), SAT("星期六");
    
    private String chinese;
    private Weekday(String chinese){
        this.chinese = chinese;
    }
    public String toChinese(){
        return chinese;
    }
}

                

class SuperClass{
    int d1;
    public SuperClass(int d1){
        this.d1=d1;
    }
}
class SubClass extends SuperClass{
    int d2;
    SubClass(int d1,int d2){
        super(d1);
        this.d2=d2;
    }
    public void ConditionCheck(){
    if(d1==10&&d2==15){
            System.out.println("Condition is true!!");
            }
    else{
            System.out.println("Condition is false!!");
    }
    }
}
public class PolymorphismTest {
    public static void main(String[] args) {
      SubClass sub=new SubClass(10,15);
      sub.ConditionCheck();
    }
  
}
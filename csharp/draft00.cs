// See https://aka.ms/new-console-template for more information
Console.Write("Hello, World!");
Console.Write("\n");
Console.WriteLine("Hello, World!");
Console.WriteLine("Hello, \"World\"!");
Console.WriteLine("The current time is " + DateTime.Now);
Console.WriteLine(@"   c:\source\repos
      (this is where your code goes)");
Console.WriteLine("\u3053\u3093\u306B\u3061\u306F World!");

Console.WriteLine('b'); //character
Console.WriteLine(123); //int
Console.WriteLine(12.3m); //decimal
Console.WriteLine(12.3M); //decimal
Console.WriteLine(true); //bool
Console.WriteLine(false); //bool

string str0;
str0 = "Bob";
string str1 = "Bob";
char userOption;
int gameScore;
decimal particlesPerMillion;
bool processedCustomer;
var message = "Hello world!"; //string

string name = "Bob";
int count = 3;
decimal temperature = 34.4m;
Console.WriteLine($"Hello, {name}! You have {count} in your inbox. The temperature is {temperature} celsius.");

string projectName = "First-Project";
Console.WriteLine($@"C:\Output\{projectName}\Data");

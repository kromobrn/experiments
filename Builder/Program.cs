using System;
using Builder;

namespace Builder
{
    public class Program
    {
        public static void Main()
        {
            var myObject = new HandlingStates.MyObjectBuilder()
                .WithPropertyA(new HandlingStates.MyProperty(1, false))
                .WithPropertyB(new HandlingStates.MyProperty(2, true))
                .WithPropertyC(new HandlingStates.MyProperty(3, false))
                .Build();
        }
    }
}

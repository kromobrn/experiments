using System;

namespace CompositeKpi
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }

    public class Constant : IValue
    {
        public int Value { get; }
        public Constant(int value) => Value = value;
    }
    public interface IValue { }

    public interface IValueProvider { IValue Evaluate(); }

    public class ConstantValueProvider : IValueProvider
    {
        public IValue Evaluate()
        {
            throw new NotImplementedException();
        }
    }
}

using System;
using System.Collections;
using System.Collections.Generic;

namespace CompositeKpi
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
        }
    }

    public interface INodeValue { }

    public abstract class NodeValue<T> : INodeValue
    {
        public T Value { get; }
        protected NodeValue(T value) => Value = value;
    }

    public class IntegerValue : NodeValue<int>
    {
        public IntegerValue(int value) : base(value) { }
    }
    public class DateTimeValue : NodeValue<DateTimeValue>
    {
        public DateTimeValue(DateTimeValue value) : base(value) { }
    }

    public interface INode<T> where T: INodeValue
    {
        public T Value { get; set; }
        public INode<T> Left { get; set; }
        public INode<T> Right { get; set; }
    }

    public class Constant : IEnumerator<int>
    {
        public int Current => 1;

        object IEnumerator.Current => Current;

        public void Dispose()
        {
            throw new NotImplementedException();
        }

        public bool MoveNext()
        {
            return true;
        }

        public void Reset()
        {

        }
    }

}

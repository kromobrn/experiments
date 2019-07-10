using System;
using System.Linq;

namespace decorator
{
    public interface IExecutor
    {
        void Execute();
    }

    public class MyObject : IExecutor
    {
        public void Execute() => Console.WriteLine("I'm being executed!");
    }

    public abstract class DecoratedExecutor : IExecutor
    {
        readonly IExecutor wrapped;
        
        public DecoratedExecutor(IExecutor executor) 
            => wrapped = executor;
        
        public virtual void Execute() => wrapped.Execute();
    }

    public class MultipleExecutor : DecoratedExecutor
    {      
        readonly int count;

        public MultipleExecutor(
            IExecutor executor, 
            int count = 1
        ) : base(executor)
            => this.count = count;

        public override void Execute()
        {
            foreach (var _ in Enumerable.Range(0, count))
                base.Execute();
        }
    }

    public class Program
    {
        public static void Main(string[] args)
        {
            var executor = new MyObject();
            executor.Execute();

            var doubleExecutor = new MultipleExecutor(executor, 2);
            doubleExecutor.Execute();
        }
    }
}

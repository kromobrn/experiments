using System;

namespace Builder.HandlingStates
{    
    public struct MyProperty
    {
        int value;
        bool option;

        public MyProperty(int value, bool option)
        {
            this.value = value;
            this.option = option;
        }
    }
    
    public class MyObject
    {
        public MyProperty _propertyA { get; }
        public MyProperty _propertyB { get; }
        public MyProperty _propertyC { get; }

        public MyObject() { }

        public MyObject(
            MyProperty propertyA,
            MyProperty propertyB,
            MyProperty propertyC
        )
        {
            _propertyA = propertyA;
            _propertyB = propertyB;
            _propertyC = propertyC;
        }
    }
    
    public interface IBuilder<T>
    {
        T Build();
        void Reset();
    }
    
    public class MyObjectBuilder : IBuilder<MyObject>
    {
        private MyProperty propertyA;
        private MyProperty propertyB;
        private MyProperty propertyC;
        
        public MyObjectBuilder()
        {
            SetDefaultValues();
        }

        private void SetDefaultValues()
        {
            propertyA = new MyProperty(0, false);
            propertyB = new MyProperty(0, false);
            propertyC = new MyProperty(0, false);
        }

        public MyObjectBuilder WithPropertyA(MyProperty property)
        {
            propertyA = property;
            return this;
        }

        public MyObjectBuilder WithPropertyB(MyProperty property)
        {
            propertyB = property;
            return this;
        }

        public MyObjectBuilder WithPropertyC(MyProperty property)
        {
            propertyC = property;
            return this;
        }

        public MyObject Build()
        {
            return new MyObject(propertyA, propertyB, propertyC);
        }

        public void Reset()
        {
            throw new NotImplementedException();
        }
    }
}

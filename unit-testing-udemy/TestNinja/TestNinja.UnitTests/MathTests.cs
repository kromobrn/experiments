using NUnit.Framework;
using TestNinja.Fundamentals;

namespace TestNinja.UnitTests
{
    [TestFixture]
    public class MathTests
    {
        Math _math;

        [SetUp]
        public void SetUp()
        {
            _math = new Math();
        }

        [Test]
        public void Add_WhenCalled_ReturnTheSumOfArguments()
        {
            // Act
            var result = _math.Add(1, 2);

            // Assert
            Assert.That(result, Is.EqualTo(3));
        }

        [Test]
        [TestCase(1, 2, 2)]
        [TestCase(2, 1, 2)]
        [TestCase(1, 1, 1)]
        public void Max_WhenCalled__ReturnTheGreaterArgument(int a, int b, int expectedResult)
        {
            // Act
            var result = _math.Max(a, b);
            
            // Assert
            Assert.That(result, Is.EqualTo(expectedResult));
        }
    }
}

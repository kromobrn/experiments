using NUnit.Framework;
using TestNinja.Fundamentals;

namespace TestNinja.UnitTests
{
    [TestFixture]
    public class FizzBuzzTests
    {
        [Test]
        public void GetOutput_NumberIsMultipleOfThreeAndFive_ReturnFizzBuzz()
        {
            var multipleOf3And5 = 3 * 5;
            var result = FizzBuzz.GetOutput(multipleOf3And5);

            Assert.That(result, Is.EqualTo("FizzBuzz"));
        }

        [Test]
        public void GetOutput_NumberIsMultipleOfThreeButNotOfFive_ReturnFizz()
        {
            var multipleOf3only = 3;
            var result = FizzBuzz.GetOutput(multipleOf3only);

            Assert.That(result, Is.EqualTo("Fizz"));
        }

        [Test]
        public void GetOutput_NumberIsMultipleOfFiveButNotOfThree_ReturnBuzz()
        {
            var multipleOf5only = 5;
            var result = FizzBuzz.GetOutput(multipleOf5only);

            Assert.That(result, Is.EqualTo("Buzz"));
        }

        [Test]
        public void GetOutput_NumberIsNotMultipleOrThreeoOrFive_ReturnNumberAsString()
        {
            var number = 2;
            var result = FizzBuzz.GetOutput(number);

            Assert.That(result, Is.EqualTo(number.ToString()));
        }
    }
}

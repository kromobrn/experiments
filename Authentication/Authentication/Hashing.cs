using System;
using System.Text;

namespace Qualyteam.g0y.Hashing
{
    public interface IHasher
    {
        byte[] Hash(byte[] byteCode);
        string Hash(string str);
    }

    public class MD5Hasher : IHasher
    {
        public string Hash(string str)
        {
            var bytes = Encoding.ASCII.GetBytes(str);
            return Encoding.ASCII.GetString(Hash(bytes));
        }

        public byte[] Hash(byte[] byteCode) => MD5(byteCode);

        private byte[] MD5(byte[] byteCode)
        {
            throw new NotImplementedException();
        }
    }

    public class SHAHasher : IHasher
    {
        public string Hash(string str)
        {
            var bytes = Encoding.ASCII.GetBytes(str);
            return Encoding.ASCII.GetString(Hash(bytes));
        }

        public byte[] Hash(byte[] byteCode) => SHA(byteCode);

        private byte[] SHA(byte[] byteCode)
        {
            throw new NotImplementedException();
        }
    }
}

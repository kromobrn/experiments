using System;
using System.Collections.Generic;
using System.Text;
using Qualyteam.g0y.Hashing;

namespace Qualyteam.g0y.Authentication
{
    /* 
        authenticator
            hashAuthenticator
                passwordAuthenticator
                    masterPasswordAuthenticator
                
    */

    public interface IUserService { }
    public class User
    {
        public string Password;
    }

    public abstract class AuthCommand { } // FingerPrintAuthCommand, QRCodeAuthCommand

    public class UserAndPasswordAuthCommand : AuthCommand
    {
        public string Username { get; private set; }
        public string Password { get; private set; }

        public UserAndPasswordAuthCommand(string username, string password)
        {
            Username = username;
            Password = password;
        }

    }

    public interface IAuthenticator
    {
        bool Authenticate();
    }

    public interface IAuthenticator<T> where T : AuthCommand
    {
        bool Authenticate(User user, T authCommand);
    }

    //public class HashAuthenticator : IAuthenticator<AuthCommand>
    //{
    //    private readonly IUserService _userService;
    //    private readonly IHasher _hasher;

    //    public HashAuthenticator(IUserService userService, IHasher hasher)
    //    {
    //        _userService = userService;
    //        _hasher = hasher;
    //    }

    //    public bool Authenticate(User user, AuthCommand authCommand)
    //    {
    //        throw new NotImplementedException();
    //    }
    //}

    public class UsernameAndPasswordAuthenticator : IAuthenticator<UserAndPasswordAuthCommand>
    {
        private readonly IUserService _userService;
        private readonly IHasher _hasher;

        public UsernameAndPasswordAuthenticator(IUserService userService, IHasher hasher)
        {
            _userService = userService;
            _hasher = hasher;
        }

        public bool Authenticate(User user, UserAndPasswordAuthCommand cmd)
        {
            return _hasher.Hash(cmd.Password).Equals(user.Password, StringComparison.OrdinalIgnoreCase);
        }
    }

    public interface ITokenService
    {
        string AuthenticateUser(string username, string password);
    }


    public class TokenService : ITokenService
    {
        //private IConfiguration _config;
        private readonly IUserService _userService;
        private readonly IAuthenticator _authenticationStrategy;

        public TokenService(
            //IConfiguration config,
            IUserService userService,
            IAuthenticator authenticationStrategy
        )
        {
            //_config = config;
            _userService = userService;
            _authenticationStrategy = authenticationStrategy;
        }

        public string AuthenticateUser(string username, string password)
        {
            var user = _userService.FindByUsername(username);

            if (user == null || !user.Active)
            {
                return null;
            }

            var passwordHash = Token.GenerateMD5Hash(password);

            foreach (var authenticationStrategy in _authenticationStrategy)
            {
                if (authenticationStrategy.CanAuthenticate(user, passwordHash))
                {
                    return GenerateToken(user);
                }
            }

            return null;
        }

        public string AuthenticateUser(User user)
        {
            if (user == null || !user.Active)
            {
                return null;
            }

            var passwordHash = Token.GenerateMD5Hash(password);

            foreach (var authenticationStrategy in _authenticationStrategy)
            {
                if (authenticationStrategy.CanAuthenticate(user, passwordHash))
                {
                    return GenerateToken(user);
                }
            }

            return null;
        }

        private string GenerateToken(User user)
        {
            /*
            var company = user.UsersCompanies.FirstOrDefault().Company;
            var userContext = new UserContext(null, user.Id, company.Id, company.Domain, user.UserName);

            return Token.BuildToken(userContext, _config["Jwt:Issuer"], _config["Jwt:Key"]);
            */
            return "";
        }
    }



    public class MasterPasswordStrategy : IAuthenticator
    {
        private readonly IIPWhiteListConfiguration _ipWhiteListConfiguration;
        private readonly string _requestIPAddress;
        private readonly string _masterPasswordHash;

        public MasterPasswordStrategy(
            IIPWhiteListConfiguration ipWhiteListConfiguration,
            string requestIPAddress,
            string masterKeyHash
        )
        {
            _ipWhiteListConfiguration = ipWhiteListConfiguration;
            _requestIPAddress = requestIPAddress;
            _masterPasswordHash = masterKeyHash;
        }

        public bool CanAuthenticate(User user, string passwordHash)
        {
            var passwordMatch = _masterPasswordHash.Equals(passwordHash, StringComparison.OrdinalIgnoreCase);
            var isAuthorizedIPAddress = _ipWhiteListConfiguration.IsAuthorizedIPAddress(_requestIPAddress);

            return passwordMatch && isAuthorizedIPAddress;
        }

    }

    public class UserCredentialStrategy : IAuthenticator
    {
        public bool CanAuthenticate(User user, string passwordHash)
        {
            return user.Password.Equals(passwordHash, StringComparison.OrdinalIgnoreCase);
        }
    }
}

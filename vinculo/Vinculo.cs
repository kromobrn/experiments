using System;

namespace Vinculos.Vinculo
{
    public class FabricaDeVinculos
    {
        public IEnumerable<IVinculo> GetVinculos(IVinculavel vinculavel)
        {

        }
    }
    
    public interface IVinculo
    {
        IFonteDeVinculos Fonte;
        IVinculavel ItemVinculado;
        
        bool PermiteInativacao();
        bool PermiteExclusao();
    }

    public abstract class Vinculo : IVinculo { }
    public class VinculoRemovivel : Vinculo { }
    public class Pendencia : Vinculo { }
    public class Dependencia : Vinculo { }

    public interface IFonteDeVinculos { }
    public class RNC : IFonteDeVinculos { }
    
    public interface IVinculavel { }
    public class Usuario : IVinculavel { }
}
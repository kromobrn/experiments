using System;

namespace Vinculos.Vinculo
{
    public interface IVinculo
    {
        void PermiteInativar();
        void PermiteExcluir();
    }
    public class VinculoRemovivel : IVinculo { }
    public class Pendencia : IVinculo { }
    public class Dependencia : IVinculo { }

    public interface IFonteDeVinculos { }
    public class RNC : IFonteDeVinculos { }
    
    public interface IVinculavel { }
    public class Usuario : IVinculavel { }
}
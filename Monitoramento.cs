using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using log4net;

namespace xiiiiii
{
    public static class StopwatchExtensions
    {
        public static T MedirExecucao<T>(this Stopwatch stopwatch, Func<T> metodo, Action<TimeSpan> callback)
        {
            try
            {
                return metodo();
            }
            finally
            {
                var tempoDecorrido = stopwatch.Elapsed;
                callback(tempoDecorrido);
            }
        }

        public static void MedirExecucao(this Stopwatch stopwatch, Action metodo, Action<TimeSpan> callback)
        {
            stopwatch.MedirExecucao(() => { metodo(); return default(object); }, callback);
        }
    }

    public class Monitoramento
    {
        private static ILog GetLogger(Type type) => LogManager.GetLogger(type);

        private static void LogarExecucao(Delegate metodo, TimeSpan tempoDecorrido, Dictionary<string, object> infoAdicional = null)
        {
            var msg = string.Format("Método: {0}.{1}\nTempo de execução(ms): {2}\n",
                metodo.Method.DeclaringType,
                metodo.Method.Name,
                tempoDecorrido.TotalMilliseconds
            );

            if (infoAdicional != null)
                msg += string.Join("\n", infoAdicional.Select(
                    p => string.Format("{0}: {1}", p.Key, p.Value.ToString()))
                );

            GetLogger(metodo.Method.DeclaringType).Info(msg);
        }

        public static T ExecutarMonitorando<T>(Func<T> metodo, Dictionary<string, object> infoAdicional = null)
        {
            return Stopwatch.StartNew().MedirExecucao(
                metodo,
                tempoDecorrido => LogarExecucao(metodo, tempoDecorrido, infoAdicional)
            );
        }
    }

    public class Programa
    {
        private object SelectAntes(string comando)
        {
            var tempoAntes = DateTime.Now;
            var sel = Select(comando);
            var duracaoQuery = DateTime.Now - tempoAntes;
            log.InfoFormat("Tempo: {1} Comando: {0}", comando, duracaoQuery.Milliseconds);
            return sel;
        }

        private object SelectDepois(string comando)
        {
            return Monitoramento.ExecutarMonitorando(
                () => Select(comando),
                new Dictionary<string, object>() { ["Consulta"] = comando }
            );
        }
    }
}

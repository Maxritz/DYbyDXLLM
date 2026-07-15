// DirectLLM CLI Harness - (C) 2026 DirectLLM Team
using System;

namespace DirectLLM.CLI
{
    public enum LogType
    {
        Info,
        Warning,
        Error,
        Debug,
        Trace
    }

    public static class TraceLogger
    {
        public static void Log(string moduleName, string message, LogType type)
        {
            var oldColor = Console.ForegroundColor;
            string stamp = DateTime.Now.ToString("HH:mm:ss.fff");

            switch (type)
            {
                case LogType.Info:
                    Console.ForegroundColor = ConsoleColor.White;
                    Console.Write($"[{stamp}][INFO] ");
                    break;
                case LogType.Warning:
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.Write($"[{stamp}][WARN] ");
                    break;
                case LogType.Error:
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.Write($"[{stamp}][ERR]  ");
                    break;
                case LogType.Debug:
                    Console.ForegroundColor = ConsoleColor.Cyan;
                    Console.Write($"[{stamp}][DEBG] ");
                    break;
                case LogType.Trace:
                    Console.ForegroundColor = ConsoleColor.DarkGray;
                    Console.Write($"[{stamp}][TRAC] ");
                    break;
            }

            Console.ForegroundColor = ConsoleColor.DarkYellow;
            Console.Write($"[{moduleName}] ");
            Console.ForegroundColor = ConsoleColor.Gray;
            Console.WriteLine(message);
            Console.ResetColor();
        }

        public static void PrintComponentsList()
        {
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine("=====================================================================");
            Console.WriteLine(" DirectLLM Core Diagnostics - Tracer Component Register");
            Console.WriteLine("=====================================================================");
            Console.ResetColor();
            Console.WriteLine("Traced Components:");
            Console.WriteLine("---------------------------------------------------------------------");
            Console.WriteLine("1. DirectXEngine   | Graphics Adapter mapping & Agility Core Fences");
            Console.WriteLine("2. ShaderCompiler  | Dynamic SM 6.6 HLSL loader & DXC shader binary link");
            Console.WriteLine("3. QuantGEMM       | INT4 Weight-Only Matrix-Mul shader dispatch runs");
            Console.WriteLine("4. KVCache         | Ring-buffer KV seq allocations & page index tracking");
            Console.WriteLine("5. Tokenizer       | Vocabulary tokenizer, subwords indexing & fallback");
            Console.WriteLine("6. WeightLoader    | INT4/INT8 Weights unpack & direct memory heaps copy");
            Console.WriteLine("7. Pipeline        | Async Command queue synchronization & loop clock");
            Console.WriteLine("---------------------------------------------------------------------");
            Console.WriteLine();
        }
    }
}

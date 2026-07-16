// DirectLLM CLI Harness - (C) 2026 DirectLLM Team
using System;
using System.Diagnostics;
using System.IO;

namespace DirectLLM.CLI
{
    class Program
    {
        static int Main(string[] args)
        {
            bool isDebug = false;
            bool isVerbose = false;
            string modelPath = "";
            string prompt = "Describe the architecture of DirectX 12 Agility SDK.";
            System.Collections.Generic.List<string> traceFilters = new System.Collections.Generic.List<string>();

            // 1. Argument parsing
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "-h" || args[i] == "--help" || args[i] == "-help")
                {
                    PrintHelp();
                    return 0;
                }
                else if (args[i] == "-debug")
                {
                    isDebug = true;
                }
                else if (args[i] == "-verbose")
                {
                    isVerbose = true;
                }
                else if (args[i] == "-trace" && i + 1 < args.Length)
                {
                    traceFilters.AddRange(args[i + 1].Split(','));
                    i++;
                }
                else if (args[i] == "--model" && i + 1 < args.Length)
                {
                    modelPath = args[i + 1];
                    i++;
                }
                else if (args[i] == "--prompt" && i + 1 < args.Length)
                {
                    prompt = args[i + 1];
                    i++;
                }
            }

            if (isDebug)
            {
                TraceLogger.PrintComponentsList();
            }

            TraceLogger.Log("CLI_Harness", "DirectLLM CLI starting up...", LogType.Info);
            TraceLogger.Log("CLI_Harness", $"Arguments loaded: Model={modelPath}, Verbose={isVerbose}, Debug={isDebug}", LogType.Debug);

            // Model path is REQUIRED. This harness drives the real native engine
            // (DybyDx.exe), which performs actual GGUF inference. There is no
            // fallback to placeholder weights or a synthetic generation loop.
            if (string.IsNullOrEmpty(modelPath))
            {
                TraceLogger.Log("CLI_Harness", "FATAL: No --model supplied. A real GGUF model is required.", LogType.Error);
                Console.Error.WriteLine("ERROR: No --model supplied. A real GGUF model is required.");
                return 1;
            }

            if (!File.Exists(modelPath))
            {
                TraceLogger.Log("CLI_Harness", $"FATAL: Model file not found: {modelPath}", LogType.Error);
                Console.Error.WriteLine($"ERROR: Model file not found: {modelPath}");
                return 1;
            }

            // Locate the native engine built from the same sources as this harness.
            string exePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "DybyDx.exe");
            if (!File.Exists(exePath))
            {
                exePath = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "DybyDx.exe"));
            }
            if (!File.Exists(exePath))
            {
                TraceLogger.Log("CLI_Harness", $"FATAL: Native engine DybyDx.exe not found (looked in {exePath}).", LogType.Error);
                Console.Error.WriteLine("ERROR: Native engine DybyDx.exe not found. Build the DybyDx target first.");
                return 1;
            }

            TraceLogger.Log("CLI_Harness", $"Launching native engine: {exePath}", LogType.Info);

            var sw = Stopwatch.StartNew();

            // Forward the user's intent to the real engine. The engine loads the
            // GGUF weights, runs genuine inference, and streams real output.
            var arguments = $"--model \"{modelPath}\" --prompt \"{prompt}\"";
            if (isVerbose) arguments += " -verbose";

            var psi = new ProcessStartInfo
            {
                FileName = exePath,
                Arguments = arguments,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            try
            {
                using (var process = Process.Start(psi))
                {
                    if (process == null)
                    {
                        TraceLogger.Log("CLI_Harness", "FATAL: Failed to start native engine.", LogType.Error);
                        return 1;
                    }

                    process.OutputDataReceived += (sender, e) =>
                    {
                        if (e.Data != null) Console.WriteLine(e.Data);
                    };
                    process.ErrorDataReceived += (sender, e) =>
                    {
                        if (e.Data != null) Console.Error.WriteLine(e.Data);
                    };

                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    process.WaitForExit();

                    sw.Stop();
                    TraceLogger.Log("CLI_Harness", $"Native engine exited in {sw.ElapsedMilliseconds} ms (code {process.ExitCode}).", LogType.Info);

                    return process.ExitCode;
                }
            }
            catch (Exception ex)
            {
                TraceLogger.Log("CLI_Harness", $"FATAL: Exception running engine: {ex.Message}", LogType.Error);
                Console.Error.WriteLine($"ERROR: {ex.Message}");
                return 1;
            }
        }

        private static void PrintHelp()
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("==========================================================");
            Console.WriteLine(" DirectLLM Local Inference Engine v1.0.0");
            Console.WriteLine("==========================================================");
            Console.ResetColor();
            Console.WriteLine("Usage: directllm.exe [options]");
            Console.WriteLine();
            Console.WriteLine("Options:");
            Console.WriteLine("  --model <path>      Path to a real GGUF model file (REQUIRED)");
            Console.WriteLine("  --prompt \"text\"     Input prompt text to feed the LLM");
            Console.WriteLine("  -debug              List all modular components and enable diagnostic logs");
            Console.WriteLine("  -verbose            Include granular tracing information for shader dispatches and KV stores");
            Console.WriteLine("  -trace <modules>    Comma-separated list of components to trace (e.g., -trace Tokenizer,QuantGEMM)");
            Console.WriteLine("  -help, -h           Show this manual");
            Console.WriteLine();
            Console.WriteLine("This harness drives the native DybyDx.exe engine, which");
            Console.WriteLine("performs actual GGUF weight loading and inference.");
        }
    }
}

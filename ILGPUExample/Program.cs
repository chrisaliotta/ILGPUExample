
using BenchmarkDotNet.Running;
using System;

namespace ILGPUExample
{
    internal class Program
    {


        private static void Main(string[] args)
        {
            //var summary = BenchmarkRunner.Run<GpuVsCpuBenchmark>();

            var a = new GpuVsCpuBenchmark();
            a.BenchmarkCpuSynchronous();
            a.BenchmarkGpuAccelerator();
            Console.WriteLine("Done");
            Console.ReadKey();
        }
    }
}
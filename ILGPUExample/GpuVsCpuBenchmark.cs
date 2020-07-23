using System;
using System.Collections.Generic;
using System.Text;
using System.Diagnostics;
using BenchmarkDotNet.Attributes;
using ILGPU;
using ILGPU.Runtime;

namespace ILGPUExample
{
    public class GpuVsCpuBenchmark
    {
        private static void MyKernel(
            Index1 index,              // The global thread index (1D in this case)
            ArrayView<float> dataView,   // A view to a chunk of memory (1D in this case)
            int constant)              // A sample uniform constant
        {
            dataView[index] = index / constant;
        }



        [Benchmark]
        public void BenchmarkGpuAccelerator()
        {
            using (var context = new Context())
            {
                foreach (var acceleratorId in Accelerator.Accelerators)
                {
                    if (acceleratorId.AcceleratorType == AcceleratorType.Cuda)
                    {
                        using var accelerator = Accelerator.Create(context, acceleratorId);
                        //Console.WriteLine($"Performing operations on {accelerator}");

                        var kernel = accelerator.LoadAutoGroupedStreamKernel
                            <Index1, ArrayView<float>, int>(MyKernel);

                        
                        using (var buffer = accelerator.Allocate<float>(1000000))
                        {
                            Stopwatch sw = new Stopwatch();
                            sw.Start();
                            // Launch buffer.Length many threads and pass a view to buffer
                            // Note that the kernel launch does not involve any boxing
                            kernel(buffer.Length, buffer.View, 42);

                            // Wait for the kernel to finish...
                            accelerator.Synchronize();
                            sw.Stop();
                            Console.WriteLine($"GPU: {sw.ElapsedTicks}");

                            // Resolve and verify data
                            var data = buffer.GetAsArray();
                            /*
                            for (int i = 0, e = data.Length; i < e; ++i)
                            {
                                if (data[i] != 42 + i)
                                    Console.WriteLine($"Error at element location {i}: {data[i]} found");
                            }
                            */
                        }
                        
                    }
                }
            }
        }

        [Benchmark]
        public void BenchmarkCpuSynchronous()
        {
            List<int> results = new List<int>(1000000);
            for (int i = 0; i < results.Count; i++)
            {
                results.Add(i + 42);
            }
        }
    }
}

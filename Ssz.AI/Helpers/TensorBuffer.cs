using System;
using static TorchSharp.torch;

namespace Ssz.AI.Helpers;

public class TensorBuffer : IDisposable
{
    public TensorBuffer(Device device, long capacity)
    {
        Device = device;
        Tensor_Buffer_Capacity = capacity; //Math.Max(totalVoxels, Tensor_Buffer_Capacity == 0 ? totalVoxels : Tensor_Buffer_Capacity * 2);

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public void EnsureCapacity(long capacity)
    {
        if (capacity <= Tensor_Buffer_Capacity)
            return;

        Tensor_device_Buffer.Dispose();
        Tensor_Cpu_Buffer.Dispose();

        Tensor_Buffer_Capacity = capacity; //Math.Max(totalVoxels, Tensor_Buffer_Capacity == 0 ? totalVoxels : Tensor_Buffer_Capacity * 2);

        Tensor_device_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: Device).DetachFromDisposeScope();
        Tensor_Cpu_Buffer = empty(new long[] { Tensor_Buffer_Capacity }, dtype: ScalarType.Float32, device: CPU).DetachFromDisposeScope();
    }

    public readonly Device Device;

    public Tensor Tensor_device_Buffer;
    public Tensor Tensor_Cpu_Buffer;
    // Текущая вместимость (в элементах), чтобы понимать, когда нужно перевыделять память
    public long Tensor_Buffer_Capacity = 0;

    public void Dispose()
    {
        Tensor_device_Buffer?.Dispose();
        Tensor_Cpu_Buffer?.Dispose();
    }
}


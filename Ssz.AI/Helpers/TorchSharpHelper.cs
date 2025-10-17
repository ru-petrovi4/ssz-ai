using Microsoft.AspNetCore.Identity;
using Ssz.Utils.Serialization;
using System.IO;
using System.Linq;
using Tensorflow;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Helpers;

public static class TorchSharpHelper
{
    public static void WriteTensor(torch.Tensor? tensor, SerializationWriter writer)
    {
        if (tensor is null)
        {
            writer.WriteArray((long[]?)null);
        }
        else
        {
            if (tensor.dtype != ScalarType.Float32)
                throw new InvalidArgumentError("tensor.dtype != ScalarType.Float32");
            writer.WriteArray(tensor.shape);
            var model = new TensorHolder(tensor);
            using (MemoryStream memoryStream = new())
            {
                model.save(memoryStream); // сериализация через поток
                writer.WriteArray(memoryStream.ToArray());
            }
        }
    }

    public static torch.Tensor? ReadTensor(SerializationReader reader)
    {
        var shape = reader.ReadArray<long>()!;
        if (shape is null)
            return null;
        long length = 1;
        foreach (int dim in shape)
            length *= dim;
        byte[] data = reader.ReadByteArray();
        var model = new TensorHolder(torch.tensor(new float[length]).reshape(shape));        
        using (MemoryStream memoryStream = new(data))
        {
            model.load(memoryStream); // загрузка данных
        }            
        return model.Tensor; // извлечь восстановленный тензор
    }

    private class TensorHolder : nn.Module
    {
        public torch.Tensor Tensor { get; set; }

        public TensorHolder(torch.Tensor tensor) : base("TensorHolder")
        {
            Tensor = tensor;
            register_buffer("tensor", tensor); // зарегистрировать для сериализации
        }
    }
}

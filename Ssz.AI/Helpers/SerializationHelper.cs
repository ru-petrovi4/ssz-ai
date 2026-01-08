using Microsoft.Extensions.Logging;
using Ssz.AI.Models;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Drawing;
using System.IO;

namespace Ssz.AI.Helpers;

public static class SerializationHelper
{
    public static void SaveToFile(string fileName, IOwnedDataSerializable ownedDataSerializable, object? context, ILogger? logger)
    {
        fileName = Path.Combine(@"Data", fileName);
        using (FileStream stream = File.Create(fileName))
        using (var writer = new SerializationWriter(stream, false))
        {
            writer.WriteOwnedDataSerializable(ownedDataSerializable, context);
        }
        logger?.LogInformation($"Saved: {fileName}");
    }

    public static void LoadFromFileIfExists(string fileName, IOwnedDataSerializable ownedDataSerializable, object? context, ILogger? logger)
    {
        fileName = Path.Combine(@"Data", fileName);
        if (File.Exists(fileName))
        {
            using (var stream = new FileStream(fileName, FileMode.Open))
            using (var reader = new SerializationReader(stream))
            {
                reader.ReadOwnedDataSerializable(ownedDataSerializable, context);
            }
        }
        logger?.LogInformation($"Loaded: {fileName}");
    }

    public static void SerializeOwnedData_DenseMatrix<T>(DenseMatrix<T?> denseMatrix, SerializationWriter writer, object? context)
        where T : class, IOwnedDataSerializable
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(denseMatrix.Dimensions);
            for (int mcy = 0; mcy < denseMatrix.Dimensions[1]; mcy += 1)
                for (int mcx = 0; mcx < denseMatrix.Dimensions[0]; mcx += 1)
                {
                    var o = denseMatrix[mcx, mcy];
                    writer.WriteOwnedDataSerializableAndRecreatable(o, context);                                      
                }
        }
    }

    public static DenseMatrix<T?> DeserializeOwnedData_DenseMatrix<T>(SerializationReader reader, object? context, Func<int, int, T> func)
        where T : class, IOwnedDataSerializable
    {        
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    var dimensions = reader.ReadArray<int>()!;
                    DenseMatrix<T?> denseMatrix = new(dimensions);                    
                    for (int mcy = 0; mcy < dimensions[1]; mcy += 1)
                        for (int mcx = 0; mcx < dimensions[0]; mcx += 1)
                        {   
                            denseMatrix[mcx, mcy] = reader.ReadOwnedDataSerializableAndRecreatable<T>(() => func(mcx, mcy), context);
                        }
                    return denseMatrix;
            }
        }
        throw new InvalidOperationException();
    }        

    public static void Write(this SerializationWriter writer, Color value)
    {
        writer.Write(value.A);
        writer.Write(value.R);
        writer.Write(value.G);
        writer.Write(value.B);
    }

    public static Color ReadColor(this SerializationReader reader)
    {
        var a = reader.ReadByte();
        var r = reader.ReadByte();
        var g = reader.ReadByte();
        var b = reader.ReadByte();
        return Color.FromArgb(a, r, g, b);
    }

}
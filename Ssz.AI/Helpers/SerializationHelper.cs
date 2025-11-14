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

    public static void SerializeOwnedData_DenseMatrix<T>(DenseMatrix<T> denseMatrix, SerializationWriter writer, object? context)
        where T : IOwnedDataSerializable
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(denseMatrix.Dimensions);
            for (int mcy = 0; mcy < denseMatrix.Dimensions[1]; mcy += 1)
                for (int mcx = 0; mcx < denseMatrix.Dimensions[0]; mcx += 1)
                {
                    var o = denseMatrix[mcx, mcy];
                    o.SerializeOwnedData(writer, context);                    
                }
        }
    }

    public static DenseMatrix<T> DeserializeOwnedData_DenseMatrix<T>(SerializationReader reader, object? context, Func<int, int, T> func)
        where T : IOwnedDataSerializable
    {        
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    var dimensions = reader.ReadArray<int>()!;
                    DenseMatrix<T> denseMatrix = new(dimensions);                    
                    for (int mcy = 0; mcy < dimensions[1]; mcy += 1)
                        for (int mcx = 0; mcx < dimensions[0]; mcx += 1)
                        {
                            var o = func(mcx, mcy);
                            o.DeserializeOwnedData(reader, context);
                            denseMatrix[mcx, mcy] = o;
                        }
                    return denseMatrix;
            }
        }
        throw new InvalidOperationException();
    }

    /// <summary>
    ///     Cannot have nulls!
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="fastList"></param>
    /// <param name="writer"></param>
    /// <param name="context"></param>
    public static void SerializeOwnedData_FastList<T>(FastList<T> fastList, SerializationWriter writer, object? context)
        where T : IOwnedDataSerializable?
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteOptimized(fastList.Count);
            for (int mcx = 0; mcx < fastList.Count; mcx += 1)
            {
                var o = fastList[mcx];                
                o!.SerializeOwnedData(writer, context);
            }
        }
    }

    /// <summary>
    ///     Do not have nulls.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="reader"></param>
    /// <param name="context"></param>
    /// <param name="func"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public static FastList<T> DeserializeOwnedData_FastList<T>(SerializationReader reader, object? context, Func<int, T> func)
        where T : IOwnedDataSerializable?
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    var count = reader.ReadOptimizedInt32()!;
                    T[] items = new T[count];
                    for (int mcx = 0; mcx < count; mcx += 1)
                    {
                        var o = func(mcx);
                        o.DeserializeOwnedData(reader, context);
                        items[mcx] = o;
                    }
                    return new FastList<T>(items);
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
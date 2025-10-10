using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace Ssz.AI.Models;

public class MatrixFloat : IOwnedDataSerializable
{
    #region construction and destruction

    public MatrixFloat(params int[] dimensions)
    {
        if (dimensions == null || dimensions.Length != 2)
            throw new ArgumentException("Размерности матрица неверны.");

        Dimensions = (int[])dimensions.Clone();
        Data = new float[dimensions[0] * dimensions[1]];
    }        

    /// <summary>
    ///     Используется только для десериализации.
    /// </summary>
    public MatrixFloat()
    {            
    }

    #endregion

    #region public functions        

    public int[] Dimensions { get; private set; } = null!;

    public float[] Data { get; set; } = null!;

    public void Fill(float value)
    {
        Array.Fill(Data, value);
    }

    public int RowsCount => Dimensions[0];
    public int ColumnsCount => Dimensions[1];
    public int TotalElementsCount => Data.Length;

    public virtual float this[int i, int j]
    {
        get => Data[i * Dimensions[1] + j];
        set => Data[i * Dimensions[1] + j] = value;
    }

    public virtual Span<float> GetRow(int i)
    {
        var columnsCount = Dimensions[1];
        return new Span<float>(Data, i * columnsCount, columnsCount);
    }

    public virtual Span<float> GetColumn(int j)
    {
        var rowsCount = Dimensions[0];
        var columnsCount = Dimensions[1];
        var column = new float[rowsCount];
        for (int i = 0; i < rowsCount; i += 1)
        {
            column[i] = Data[i * columnsCount + j];
        }
        return new Span<float>(column);
    }

    public virtual (int, int) GetIndices(int dataIndex)
    {
        var columnsCount = Dimensions[1];
        return (dataIndex % columnsCount, dataIndex / columnsCount);
    }

    public MatrixFloat Clone()
    {
        var clone = new MatrixFloat((int[])Dimensions.Clone());
        Array.Copy(Data, clone.Data, Data.Length);
        return clone;
    }

    public void Clear()
    {
        Array.Clear(Data, 0, Data.Length);
    }

    public override string ToString()
    {
        return $"MatrixFloat({string.Join(", ", Dimensions)})";
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArray(Dimensions);
            writer.WriteArrayOfSingle(Data);                
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            {
                case 1:
                    Dimensions = reader.ReadArray<int>()!;
                    Data = reader.ReadArrayOfSingle()!;
                    break;
            }
        }
    }

    public static MatrixFloat FromTensor(torch.Tensor tensor)
    {
        var data = tensor.data<float>();        
        var result = new MatrixFloat((int)tensor.shape[0], (int)tensor.shape[1]);
        data.CopyTo(result.Data);
        return result;
    }

    public torch.Tensor ToTensor(Device device)
    {
        return tensor(Data, device: device)
                .reshape(Dimensions[0], Dimensions[1]);
    }

    #endregion
}

public class MatrixFloat_ColumnMajor : MatrixFloat
{
    #region construction and destruction

    public MatrixFloat_ColumnMajor(params int[] dimensions) :
        base(dimensions)
    {
    }

    /// <summary>
    ///     Используется только для десериализации.
    /// </summary>
    public MatrixFloat_ColumnMajor()
    {
    }

    #endregion

    #region public functions    

    public override float this[int i, int j]
    {
        get => Data[i + j * Dimensions[0]];
        set => Data[i + j * Dimensions[0]] = value;
    }

    public override Span<float> GetRow(int i)
    {
        var rowsCount = Dimensions[0];
        var columnsCount = Dimensions[1];
        var row = new float[columnsCount];
        for (int j = 0; j < columnsCount; j += 1)
        {
            row[j] = Data[i + j * rowsCount];
        }
        return new Span<float>(row);
    }

    public override Span<float> GetColumn(int j)
    {
        var rowsCount = Dimensions[0];
        return new Span<float>(Data, j * rowsCount, rowsCount);
    }

    public override (int, int) GetIndices(int dataIndex)
    {
        var rowsCount = Dimensions[0];
        return (dataIndex % rowsCount, dataIndex / rowsCount);
    }    

    public new MatrixFloat_ColumnMajor Clone()
    {
        var clone = new MatrixFloat_ColumnMajor((int[])Dimensions.Clone());
        Array.Copy(Data, clone.Data, Data.Length);
        return clone;
    }

    public override string ToString()
    {
        return $"MatrixFloat_ColumnMajor({string.Join(", ", Dimensions)})";
    }

    #endregion        
}


//TensorPrimitives

//public class MatrixFloat
//{
//    #region construction and destruction

//    public MatrixFloat(params int[] dimensions)
//    {
//        Dimensions = dimensions;
//        Data = new float[dimensions[0] * dimensions[1]];
//    }

//    #endregion

//    #region public functions        

//    public int[] Dimensions { get; private set; } = null!;

//    public float[] Data { get; set; } = null!;

//    public float this[int i, int j]
//    {
//        get => Data[i + j * Dimensions[0]];
//        set => Data[i + j * Dimensions[0]] = value;
//    }

//    public Span<float> GetColumn(int j)
//    {
//        var d = Dimensions[0];
//        return new Span<float>(Data, j * d, d);
//    }

//    public MatrixFloat Clone()
//    {
//        var clone = new MatrixFloat(Dimensions);
//        Array.Copy(Data, clone.Data, Data.Length);
//        return clone;
//    }

//    public void Clear()
//    {
//        Array.Clear(Data, 0, Data.Length);
//    }
//}
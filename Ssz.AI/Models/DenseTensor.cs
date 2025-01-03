﻿using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class DenseTensor<T> : IOwnedDataSerializable
    {
        #region construction and destruction

        public DenseTensor(params int[] dimensions)
        {
            if (dimensions == null || dimensions.Length == 0)
                throw new ArgumentException("Размерности тензора не могут быть пустыми.");

            Dimensions = dimensions;
            Data = new T[GetTotalSize(dimensions)];
        }        

        public DenseTensor(T[] data)
        {
            Dimensions = [ data.Length ];
            Data = data;
        }

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public DenseTensor()
        {            
        }

        #endregion

        #region public functions

        public int[] Dimensions { get; private set; } = null!;

        public T[] Data { get; private set; } = null!;

        public T this[params int[] indices]
        {
            get => Data[GetFlatIndex(indices)];
            set => Data[GetFlatIndex(indices)] = value;
        }

        public Span<T> GetColumn(int j)
        {
            var d = Dimensions[0];
            return new Span<T>(Data, j * d, d);
        }        

        public DenseTensor<T> Clone()
        {
            var clone = new DenseTensor<T>(Dimensions);
            Array.Copy(Data, clone.Data, Data.Length);
            return clone;
        }

        public override string ToString()
        {
            return $"DenseTensor<{typeof(T).Name}>({string.Join(", ", Dimensions)})";
        }

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            using (writer.EnterBlock(1))
            {
                writer.WriteArray(Dimensions);
                writer.WriteArray(Data);                
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
                        Data = reader.ReadArray<T>()!;
                        break;
                }
            }
        }

        #endregion

        #region private functions

        private int GetFlatIndex(int[] indices)
        {
            if (indices.Length != Dimensions.Length)
                throw new ArgumentException("Количество индексов должно совпадать с размерностями тензора.");

            int flatIndex = 0;
            int multiplier = 1;
            for (int i = 0; i < Dimensions.Length; i += 1)
            {
                flatIndex += indices[i] * multiplier;
                multiplier *= Dimensions[i];
            }
            return flatIndex;
        }

        private static int GetTotalSize(int[] dimensions)
        {
            int size = 1;
            foreach (var dim in dimensions)
                size *= dim;
            return size;
        }        

        #endregion
    }


    //public IEnumerable<int[]> Indices()
    //    {
    //        var current = new int[Dimensions.Length];
    //        while (true)
    //        {
    //            yield return (int[])current.Clone();
    //            int axis = Dimensions.Length - 1;
    //            while (axis >= 0)
    //            {
    //                current[axis]++;
    //                if (current[axis] < Dimensions[axis])
    //                    break;
    //                current[axis] = 0;
    //                axis--;
    //            }

    //            if (axis < 0)
    //                break;
    //        }
    //    }

    //public class DenseTensor<T>
    //{
    //    private readonly T[] _data;
    //    public int[] Dimensions { get; }
    //    public int Length => _data.Length;

    //    public DenseTensor(int[] dimensions)
    //    {
    //        if (dimensions == null || dimensions.Length == 0)
    //            throw new ArgumentException("Размерности тензора не могут быть пустыми.");

    //        Dimensions = dimensions;
    //        _data = new T[GetTotalSize(dimensions)];
    //    }

    //    public T this[int i, int j]
    //    {
    //        get => _data[GetIndex(i, j)];
    //        set => _data[GetIndex(i, j)] = value;
    //    }

    //    public T this[int i]
    //    {
    //        get => _data[i];
    //        set => _data[i] = value;
    //    }

    //    public T[] Buffer => _data;

    //    public DenseTensor<T> Clone()
    //    {
    //        var clone = new DenseTensor<T>(Dimensions);
    //        Array.Copy(_data, clone._data, _data.Length);
    //        return clone;
    //    }

    //    private int GetIndex(int i, int j)
    //    {
    //        if (Dimensions.Length != 2)
    //            throw new InvalidOperationException("Тензор не является двухмерным.");

    //        return i * Dimensions[1] + j;
    //    }

    //    private static int GetTotalSize(int[] dimensions)
    //    {
    //        int size = 0;
    //        foreach (var dim in dimensions)
    //            size *= dim;
    //        return size;
    //    }

    //    public override string ToString()
    //    {
    //        return $"DenseTensor<{typeof(T).Name}>({string.Join(", ", Dimensions)})";
    //    }
    //}
}
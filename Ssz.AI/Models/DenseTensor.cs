using System;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class DenseTensor<T>
    {
        #region construction and destruction

        public DenseTensor(int[] dimensions)
        {
            if (dimensions == null || dimensions.Length == 0)
                throw new ArgumentException("Размерности тензора не могут быть пустыми.");

            Dimensions = dimensions;
            _data = new T[GetTotalSize(dimensions)];
        }        

        public DenseTensor(T[] data)
        {
            Dimensions = [ data.Length ];
            _data = data;
        }

        #endregion

        #region public functions

        public int[] Dimensions { get; }

        public int Length => _data.Length;        

        public T this[params int[] indices]
        {
            get => _data[GetFlatIndex(indices)];
            set => _data[GetFlatIndex(indices)] = value;
        }

        public T[] Buffer => _data;

        public DenseTensor<T> Clone()
        {
            var clone = new DenseTensor<T>(Dimensions);
            Array.Copy(_data, clone._data, _data.Length);
            return clone;
        }

        public override string ToString()
        {
            return $"DenseTensor<{typeof(T).Name}>({string.Join(", ", Dimensions)})";
        }

        #endregion

        #region private functions

        private int GetFlatIndex(int[] indices)
        {
            if (indices.Length != Dimensions.Length)
                throw new ArgumentException("Количество индексов должно совпадать с размерностями тензора.");

            int flatIndex = 0;
            int multiplier = 1;
            for (int i = Dimensions.Length - 1; i >= 0; i--)
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

        #region private fields

        private readonly T[] _data;        

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
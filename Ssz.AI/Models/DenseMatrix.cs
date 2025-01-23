﻿using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class DenseMatrix<T> : IOwnedDataSerializable
    {
        #region construction and destruction

        public DenseMatrix(params int[] dimensions)
        {
            if (dimensions == null || dimensions.Length != 2)
                throw new ArgumentException("Размерности матрица неверны.");

            Dimensions = dimensions;
            Data = new T[dimensions[0] * dimensions[1]];
        }        

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public DenseMatrix()
        {
        }

        #endregion

        #region public functions

        public int[] Dimensions { get; private set; } = null!;

        public T[] Data { get; private set; } = null!;

        public T this[params int[] indices]
        {
            get => Data[indices[0] + indices[1] * Dimensions[0]];
            set => Data[indices[0] + indices[1] * Dimensions[0]] = value;
        }

        public DenseMatrix<T> Clone()
        {
            var clone = new DenseMatrix<T>(Dimensions);
            Array.Copy(Data, clone.Data, Data.Length);
            return clone;
        }

        public override string ToString()
        {
            return $"DenseMatrix<{typeof(T).Name}>({string.Join(", ", Dimensions)})";
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
    }
}
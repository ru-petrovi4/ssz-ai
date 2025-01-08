using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class MatrixFloat : IOwnedDataSerializable
    {
        #region construction and destruction

        public MatrixFloat(params int[] dimensions)
        {
            if (dimensions == null || dimensions.Length != 2)
                throw new ArgumentException("Размерности матрица неверны.");

            Dimensions = dimensions;
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

        public float[] Data { get; private set; } = null!;

        public float this[int i, int j]
        {
            get => Data[i + j * Dimensions[0]];
            set => Data[i + j * Dimensions[0]] = value;
        }

        public Span<float> GetColumn(int j)
        {
            var d = Dimensions[0];
            return new Span<float>(Data, j * d, d);
        }        

        public MatrixFloat Clone()
        {
            var clone = new MatrixFloat(Dimensions);
            Array.Copy(Data, clone.Data, Data.Length);
            return clone;
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

        #endregion        
    }
}
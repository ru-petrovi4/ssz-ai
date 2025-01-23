using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;

namespace Ssz.AI.Models
{
    public class VectorFloat : MatrixFloat
    {
        #region construction and destruction

        public VectorFloat(int dimension) :
            base([ dimension, 1 ])
        {            
        }        

        /// <summary>
        ///     Используется только для десериализации.
        /// </summary>
        public VectorFloat() :
            base()
        {            
        }

        #endregion

        #region public functions                

        public float this[int i]
        {
            get => Data[i];
            set => Data[i] = value;
        }             

        public new VectorFloat Clone()
        {
            var clone = new VectorFloat(Dimensions[0]);
            Array.Copy(Data, clone.Data, Data.Length);
            return clone;
        }

        public override string ToString()
        {
            return $"VectorFloat({Dimensions[0]})";
        }        

        #endregion        
    }
}
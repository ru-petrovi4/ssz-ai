using System;

namespace Ssz.AI.Models;

public class Vector3DFloat : VectorFloat
{
    #region construction and destruction

    public Vector3DFloat() :
        base(3)
    {
    }        

    #endregion

    #region public functions                

    public float X
    {
        get => Data[0];
        set => Data[0] = value;
    }

    public float Y
    {
        get => Data[1];
        set => Data[1] = value;
    }

    public float Z
    {
        get => Data[2];
        set => Data[2] = value;
    }

    public new Vector3DFloat Clone()
    {
        var clone = new Vector3DFloat();
        Array.Copy(Data, clone.Data, Data.Length);
        return clone;
    }

    public override string ToString()
    {
        return $"Vector3DFloat";
    }

    #endregion
}

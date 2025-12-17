using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System;
using System.Drawing;

namespace Ssz.AI.Models.CortexVisualisationModel;

public class InputItem : IOwnedDataSerializable
{
    public int Index;

    /// <summary>
    /// [-pi, pi)
    /// </summary>
    public float Angle;

    /// <summary>
    /// 
    /// </summary>
    public float Magnitude;

    public Color Color;

    public float SimilarityThreshold = Single.MinValue;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Index);
        writer.Write(Angle);
        writer.Write(Magnitude);
        writer.Write(Color);
        writer.Write(SimilarityThreshold);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Index = reader.ReadInt32();
        Angle = reader.ReadSingle();
        Magnitude = reader.ReadSingle();
        Color = reader.ReadColor();
        SimilarityThreshold = reader.ReadSingle();
    }

    public override string ToString()
    {
        return $"Angle: {Angle:F1}; Magnitude: {Magnitude:F03}";
    }
}

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

    public int Main_MiniColumnIndex;

    public float X_Retina;

    public float Y_Retina;

    public int HyperColumnCenter_MiniColumnIndex;

    public float X_HyperColumnCenter_Retina;

    public float Y_HyperColumnCenter_Retina;

    public Color Color;

    /// <summary>
    ///     Distance from center in ideal pinwheel in minicolumns
    /// </summary>
    public float DistanceFromCenter = Single.MinValue;    

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Index);
        writer.Write(Angle);
        writer.Write(Magnitude);
        writer.Write(Main_MiniColumnIndex);
        writer.Write(X_Retina);
        writer.Write(Y_Retina);
        writer.Write(HyperColumnCenter_MiniColumnIndex);
        writer.Write(X_HyperColumnCenter_Retina);
        writer.Write(Y_HyperColumnCenter_Retina);
        writer.Write(Color);
        writer.Write(DistanceFromCenter);        
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Index = reader.ReadInt32();
        Angle = reader.ReadSingle();
        Magnitude = reader.ReadSingle();
        Main_MiniColumnIndex = reader.ReadInt32();
        X_Retina = reader.ReadSingle();
        Y_Retina = reader.ReadSingle();
        HyperColumnCenter_MiniColumnIndex = reader.ReadInt32();
        X_HyperColumnCenter_Retina = reader.ReadSingle();
        Y_HyperColumnCenter_Retina = reader.ReadSingle();
        Color = reader.ReadColor();
        DistanceFromCenter = reader.ReadSingle();        
    }

    public override string ToString()
    {
        return $"Angle: {Angle:F1}; Magnitude: {Magnitude:F03}; XRetina: {X_Retina:F03}; YRetina: {Y_Retina:F03}";
    }
}

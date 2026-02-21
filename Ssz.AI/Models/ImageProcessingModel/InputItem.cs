using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System;
using System.Drawing;

namespace Ssz.AI.Models.ImageProcessingModel;

public class InputItem : IOwnedDataSerializable
{
    public int Index;

    /// <summary>
    /// [-pi, pi)
    /// </summary>
    public float GradientAngle;

    /// <summary>
    /// 
    /// </summary>
    public float GradientMagnitude;

    public int MainRetinaXYAngle_MiniColumnIndex;

    public float RetinaXAngle;

    public float RetinaYAngle;

    public int HyperColumnCenter_MiniColumnIndex;

    public float HyperColumnCenter_RetinaXAngle;

    public float HyperColumnCenter_RetinaYAngle;

    public Color GradientAngleMagnitude_Color;

    public Color RetinaXYAngle_Color;

    /// <summary>
    ///     Distance from center in ideal pinwheel in minicolumns
    /// </summary>
    public float DistanceFromCenter = Single.MinValue;    

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Index);
        writer.Write(GradientAngle);
        writer.Write(GradientMagnitude);
        writer.Write(MainRetinaXYAngle_MiniColumnIndex);
        writer.Write(RetinaXAngle);
        writer.Write(RetinaYAngle);
        writer.Write(HyperColumnCenter_MiniColumnIndex);
        writer.Write(HyperColumnCenter_RetinaXAngle);
        writer.Write(HyperColumnCenter_RetinaYAngle);
        writer.Write(GradientAngleMagnitude_Color);
        writer.Write(RetinaXYAngle_Color);
        writer.Write(DistanceFromCenter);        
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Index = reader.ReadInt32();
        GradientAngle = reader.ReadSingle();
        GradientMagnitude = reader.ReadSingle();
        MainRetinaXYAngle_MiniColumnIndex = reader.ReadInt32();
        RetinaXAngle = reader.ReadSingle();
        RetinaYAngle = reader.ReadSingle();
        HyperColumnCenter_MiniColumnIndex = reader.ReadInt32();
        HyperColumnCenter_RetinaXAngle = reader.ReadSingle();
        HyperColumnCenter_RetinaYAngle = reader.ReadSingle();
        GradientAngleMagnitude_Color = reader.ReadColor();
        RetinaXYAngle_Color = reader.ReadColor();
        DistanceFromCenter = reader.ReadSingle();

        // 
        //ColorAngleMagnitude = Visualisation.ColorFromHSV(ColorAngleMagnitude.GetHue() / 360.0f, 1.0, 1.0);
    }

    public override string ToString()
    {
        return $"Angle: {GradientAngle:F1}; Magnitude: {GradientMagnitude:F03}; XRetina: {RetinaXAngle:F03}; YRetina: {RetinaYAngle:F03}";
    }
}

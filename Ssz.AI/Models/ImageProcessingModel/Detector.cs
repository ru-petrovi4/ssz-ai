using Ssz.AI.Helpers;
using Ssz.AI.Models.Primitives;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Numerics;

namespace Ssz.AI.Models.ImageProcessingModel;

public abstract class Detector : IOwnedDataSerializable
{
    public Retina Retina = null!;

    public int DI;

    public int DJ;

    public double CenterXPixels;

    public double CenterYPixels;    

    public int BitIndexInHash;

    public bool Temp_IsActivated;

    public int Temp_IsActivatedCount;

    public float Temp_Density;

    public FastList<RetinaPoint> Temp_RetinaPoints = null!;

    public FastList<FeaturesVectorSample> Temp_FeaturesVectorSamples = null!;

    public abstract void SerializeOwnedData(SerializationWriter writer, object? context);

    public abstract void DeserializeOwnedData(SerializationReader reader, object? context);

    /// <summary>
    ///     Precondition: !!! Gradient in Temp_RetinaPoints must be calculated !!!
    /// </summary>
    public abstract bool CalculateIsActivated();
}

public class SimpleDetector : Detector
{
    /// <summary>
    ///     Detecting value average.
    /// </summary>
    public float Average;

    public override void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Average);
        writer.Write(BitIndexInHash);
    }

    public override void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Average = reader.ReadSingle();
        BitIndexInHash = reader.ReadInt32();
    }

    /// <summary>
    ///     Precondition: !!! Gradient in Temp_RetinaPoints must be calculated !!!
    /// </summary>
    public override bool CalculateIsActivated()
    {
        for (int rp_Index = 0; rp_Index < Temp_RetinaPoints.Count; rp_Index += 1)
        {
            bool activated = CalculateIsActivated(ref Temp_RetinaPoints[rp_Index].FeaturesVector);
            if (activated)
                return true;
        }
        return false;
    }

    public bool CalculateIsActivated(ref FeaturesVector featuresVector)
    {
        return false;

        //if (gradientInPoint.Magnitude < Retina.Constants.MinGradientMagnitudeInclusive ||
        //        gradientInPoint.Magnitude >= Retina.Constants.MaxGradientMagnitudeExclusive)
        //    return false;

        //GradientRange detectorGradientRange = Retina.DetectorGradientRanges[(int)gradientInPoint.Magnitude, (int)MathHelper.RadiansToDegrees((float)gradientInPoint.Angle)]!;

        //bool activated = Average >= gradientMagnitude_DetectorValueRange.LowerInclusive &&
        //    Average < gradientMagnitude_DetectorValueRange.UpperExclusive;
        //if (activated)
        //    return true;

        //// [-pi, pi)
        //float gradientAngleMinInclusive = gradientAngle_DetectorValueRange.LowerInclusive;
        //float gradientAngleMaxExclusive = gradientAngle_DetectorValueRange.UpperExclusive;
        //if (MathF.Abs(gradientAngleMinInclusive - gradientAngleMaxExclusive) < MathF.PI / 180)
        //    return true;

        //if (gradientAngleMaxExclusive > gradientAngleMinInclusive)
        //    activated = (GradientAngle_Average >= gradientAngleMinInclusive) && (GradientAngle_Average < gradientAngleMaxExclusive);
        //else
        //    activated = (GradientAngle_Average >= gradientAngleMinInclusive) || (GradientAngle_Average < gradientAngleMaxExclusive);
        //return activated;
    }
}

public class GradientComplexDetector : Detector
{
    /// <summary>
    ///     [0, Constants.MaxGradientMagnitudeInclusive)
    /// </summary>
    public float GradientMagnitude_Average;

    /// <summary>
    ///     [-pi, pi)
    /// </summary>
    public float GradientAngle_Average;

    public override void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(GradientMagnitude_Average);
        writer.Write(GradientAngle_Average);
        writer.Write(BitIndexInHash);
    }

    public override void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        GradientMagnitude_Average = reader.ReadSingle();
        GradientAngle_Average = reader.ReadSingle();
        BitIndexInHash = reader.ReadInt32();
    }

    /// <summary>
    ///     Precondition: !!! Gradient in Temp_RetinaPoints must be calculated !!!
    /// </summary>
    public override bool CalculateIsActivated()
    {
        for (int rp_Index = 0; rp_Index < Temp_RetinaPoints.Count; rp_Index += 1)
        {
            bool activated = CalculateIsActivated(ref Temp_RetinaPoints[rp_Index].FeaturesVector);
            if (activated)
                return true;
        }
        return false;
    }

    /// <summary>
    /// Активация по AND
    /// </summary>
    /// <param name="gradientInPoint"></param>
    /// <returns></returns>
    public bool CalculateIsActivated(ref FeaturesVector featuresVector)
    {
        float gradientMagnitude = featuresVector[FeaturesVector.GradientMagnitude_Index];        

        if (gradientMagnitude < Retina.Constants.MinGradientMagnitudeInclusive ||
                gradientMagnitude >= Retina.Constants.MaxGradientMagnitudeExclusive)
            return false;

        float gradientAngle = featuresVector[FeaturesVector.GradientAngle_Index];

        int gradientMagnitude_AsIndex =  (int)gradientMagnitude;
        int gradientAngle_AsIndex = (int)MathHelper.RadiansToDegrees((float)gradientAngle);
        DetectorValueRange gradientMagnitude_DetectorValueRange = Retina.GradientMagnitude_DetectorValueRanges[gradientMagnitude_AsIndex, gradientAngle_AsIndex]!;
        DetectorValueRange gradientAngle_DetectorValueRange = Retina.GradientAngle_DetectorValueRanges[gradientMagnitude_AsIndex, gradientAngle_AsIndex]!;

        bool activated = GradientMagnitude_Average >= gradientMagnitude_DetectorValueRange.LowerInclusive &&
            GradientMagnitude_Average < gradientMagnitude_DetectorValueRange.UpperExclusive;
        if (activated)
            return true;

        // [-pi, pi)
        float gradientAngleMinInclusive = gradientAngle_DetectorValueRange.LowerInclusive;
        float gradientAngleMaxExclusive = gradientAngle_DetectorValueRange.UpperExclusive;
        if (MathF.Abs(gradientAngleMinInclusive - gradientAngleMaxExclusive) < MathF.PI / 180)
            return true;

        if (gradientAngleMaxExclusive > gradientAngleMinInclusive)
            activated = (GradientAngle_Average >= gradientAngleMinInclusive) && (GradientAngle_Average < gradientAngleMaxExclusive);
        else
            activated = (GradientAngle_Average >= gradientAngleMinInclusive) || (GradientAngle_Average < gradientAngleMaxExclusive);
        return activated;
    }

    public bool GetIsActivated_Obsolete(GradientInPoint[,] gradientMatrix, IConstantsObsolete constants, Vector2 offset = default)
    {
        (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(CenterXPixels - offset.X, CenterYPixels - offset.Y, gradientMatrix);

        if (magnitude < constants.MinGradientMagnitudeInclusive)
            return false;

        //bool activated = (magnitude >= GradientMagnitudeLowLimit) && (magnitude < GradientMagnitudeMax);
        //if (!activated)
        //    return false;

        //if (GradientAngleMax > gradientAngleMin)
        //    activated = (angle >= gradientAngleMin) && (angle < GradientAngleMax);
        //else
        //    activated = (angle >= gradientAngleMin) || (angle < GradientAngleMax);
        return false;
    }
}

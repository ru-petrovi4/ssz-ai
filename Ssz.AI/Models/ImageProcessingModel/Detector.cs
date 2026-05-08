using Ssz.AI.Helpers;
using Ssz.AI.Models.Primitives;
using Ssz.Utils;
using Ssz.Utils.Serialization;
using System;
using System.Numerics;

namespace Ssz.AI.Models.ImageProcessingModel;

public abstract class Detector : IOwnedDataSerializable
{
    public DetectingPoint DetectingPoint = null!;

    public int BitIndexInHash;

    public bool Temp_IsActivated;    

    public abstract void SerializeOwnedData(SerializationWriter writer, object? context);

    public abstract void DeserializeOwnedData(SerializationReader reader, object? context);

    /// <summary>
    ///     Precondition: !!! Gradient in Temp_RetinaPoints must be calculated !!!
    /// </summary>
    public abstract bool CalculateIsActivated();
}

public class SimpleDetector : Detector
{
    #region construction and destruction

    public SimpleDetector(DetectingPoint detectingPoint, int featuresVector_Index)
    {
        DetectingPoint = detectingPoint;
        FeaturesVector_Index = featuresVector_Index;
    }

    #endregion

    public readonly int FeaturesVector_Index;

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
        for (int rp_Index = 0; rp_Index < DetectingPoint.Temp_RetinaPoints.Count; rp_Index += 1)
        {
            bool activated = CalculateIsActivated(ref DetectingPoint.Temp_RetinaPoints[rp_Index].FeaturesVector);
            if (activated)
                return true;
        }
        return false;
    }

    private bool CalculateIsActivated(ref FeaturesVector featuresVector)
    {
        float value = featuresVector[FeaturesVector_Index];

        if (FeaturesVector_Index == FeaturesVector.GradientMagnitude_Index)
        {
            if (value < DetectingPoint.Retina.Constants.MinGradientMagnitudeInclusive ||
                    value >= DetectingPoint.Retina.Constants.MaxGradientMagnitudeExclusive)
                return false;

            int gradientMagnitude_AsIndex = (int)value;
            int gradientAngle_AsIndex = DetectingPoint.Retina.GradientMagnitude_DetectorValueRanges.Dimensions[1] / 2;
            DetectorValueRange gradientMagnitude_DetectorValueRange = DetectingPoint.Retina.GradientMagnitude_DetectorValueRanges[gradientMagnitude_AsIndex, gradientAngle_AsIndex]!;

            return Average >= gradientMagnitude_DetectorValueRange.LowerInclusive &&
                Average < gradientMagnitude_DetectorValueRange.UpperExclusive;
        }
        else if (FeaturesVector_Index == FeaturesVector.GradientAngle_Index)
        {
            int gradientMagnitude_AsIndex = DetectingPoint.Retina.GradientAngle_DetectorValueRanges.Dimensions[0] / 2;
            int gradientAngle_AsIndex = (int)MathHelper.RadiansToDegrees((float)value);
            DetectorValueRange gradientAngle_DetectorValueRange = DetectingPoint.Retina.GradientAngle_DetectorValueRanges[gradientMagnitude_AsIndex, gradientAngle_AsIndex]!;

            // [-pi, pi)
            float gradientAngleMinInclusive = gradientAngle_DetectorValueRange.LowerInclusive;
            float gradientAngleMaxExclusive = gradientAngle_DetectorValueRange.UpperExclusive;
            if (MathF.Abs(gradientAngleMinInclusive - gradientAngleMaxExclusive) < MathF.PI / 180)
                return true;

            bool activated;
            if (gradientAngleMaxExclusive > gradientAngleMinInclusive)
                activated = (Average >= gradientAngleMinInclusive) && (Average < gradientAngleMaxExclusive);
            else
                activated = (Average >= gradientAngleMinInclusive) || (Average < gradientAngleMaxExclusive);
            return activated;
        }

        return false;
    }
}

public class GradientComplexDetector : Detector
{
    #region construction and destruction

    public GradientComplexDetector(DetectingPoint detectingPoint)
    {
        DetectingPoint = detectingPoint;
    }

    #endregion

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
        for (int rp_Index = 0; rp_Index < DetectingPoint.Temp_RetinaPoints.Count; rp_Index += 1)
        {
            bool activated = CalculateIsActivated(ref DetectingPoint.Temp_RetinaPoints[rp_Index].FeaturesVector);
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
    private bool CalculateIsActivated(ref FeaturesVector featuresVector)
    {
        float gradientMagnitude = featuresVector[FeaturesVector.GradientMagnitude_Index];        

        if (gradientMagnitude < DetectingPoint.Retina.Constants.MinGradientMagnitudeInclusive ||
                gradientMagnitude >= DetectingPoint.Retina.Constants.MaxGradientMagnitudeExclusive)
            return false;

        float gradientAngle = featuresVector[FeaturesVector.GradientAngle_Index];

        int gradientMagnitude_AsIndex =  (int)gradientMagnitude;
        int gradientAngle_AsIndex = (int)MathHelper.RadiansToDegrees((float)gradientAngle);
        DetectorValueRange gradientMagnitude_DetectorValueRange = DetectingPoint.Retina.GradientMagnitude_DetectorValueRanges[gradientMagnitude_AsIndex, gradientAngle_AsIndex]!;
        DetectorValueRange gradientAngle_DetectorValueRange = DetectingPoint.Retina.GradientAngle_DetectorValueRanges[gradientMagnitude_AsIndex, gradientAngle_AsIndex]!;

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
        (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(DetectingPoint.CenterXPixels - offset.X, DetectingPoint.CenterYPixels - offset.Y, gradientMatrix);

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

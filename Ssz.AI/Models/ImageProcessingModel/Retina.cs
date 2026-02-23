using Microsoft.Extensions.DependencyInjection;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Numerics.Tensors;
using static Ssz.AI.Models.Cortex_Simplified;

namespace Ssz.AI.Models.ImageProcessingModel;

public class Retina : ISerializableModelObject
{
    #region construction and destruction

    public Retina(IRetinaConstants constants)
    {
        float retinaDetectorsDeltaPixels = constants.RetinaDetectorsDeltaPixels;
        Detectors = new DenseMatrix<Detector>(
            (int)Math.Round(constants.RetinaImagePixelSize.Width / retinaDetectorsDeltaPixels, 0), 
            (int)Math.Round(constants.RetinaImagePixelSize.Height / retinaDetectorsDeltaPixels, 0));
        foreach (int dJ in Enumerable.Range(0, Detectors.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, Detectors.Dimensions[0]))
            {
                Detector detector = new()
                {
                    DI = dI,
                    DJ = dJ,
                    CenterXPixels = dI * retinaDetectorsDeltaPixels,
                    CenterYPixels = dJ * retinaDetectorsDeltaPixels,
                };
                Detectors[dI, dJ] = detector;
            }
    }

    #endregion

    #region public functions

    public DenseMatrix<Detector> Detectors;

    /// <summary>
    ///     Диапазоны для детекторов в зависимости от модуля градиента и угла градиента (в градусах).
    /// </summary>
    public DenseMatrix<DetectorRange> DetectorsRanges = new();

    //public float[] AngleAccumulativeDistribution = null!;

    /// <summary>
    ///     Generates model data after construction.
    /// </summary>
    public void GenerateOwnedData(Random initializationRandom, IRetinaConstants constants, GradientDistribution gradientDistribution)
    {
        // TODO gradientDistribution -> DetectorRanges            
        int gradientMagnitudeRange = constants.GeneratedMaxGradientMagnitude / constants.MagnitudeRangesCount;

        DetectorsRanges = new DenseMatrix<DetectorRange>(constants.GeneratedMaxGradientMagnitude + gradientMagnitudeRange, 360);                                  

        float gmIn1 = (constants.GeneratedMaxGradientMagnitude - constants.GeneratedMinGradientMagnitude) / constants.HyperColumnDefinedRadius_MiniColumns;

        //float angleRange0 = MathF.Atan2(constants.K5, constants.AngleRangeDegree_LimitMagnitude / gmIn1) * 4.0f;            
        //float angleRange0 = constants.AngleRangeDegreeMin * MathF.PI / 180;
        //float angleRange1 = constants.AngleRangeDegreeMax * MathF.PI / 180;
        foreach (int gradientMagnitudeIdx in Enumerable.Range(0, DetectorsRanges.Dimensions[0]))
        {
            int gradientMagnitude = gradientMagnitudeIdx;
            if (gradientMagnitude < constants.GeneratedMinGradientMagnitude)
                gradientMagnitude = constants.GeneratedMinGradientMagnitude;
            float angleRange;
            //if (gradientMagnitude < constants.AngleRangeDegree_LimitMagnitude)
            //    angleRange = angleRange0 + (angleRange1 - angleRange0) * (constants.AngleRangeDegree_LimitMagnitude - gradientMagnitude) / (constants.AngleRangeDegree_LimitMagnitude - constants.GeneratedMinGradientMagnitude);
            //else
                //angleRange = angleRange0;
            angleRange = MathF.Atan2(1.8f, gradientMagnitude / gmIn1) * 4.0f; // constants.K5

            if (angleRange > 2 * MathF.PI)
                angleRange = 2 * MathF.PI;

            foreach (int gradientAngleDegreeIdx in Enumerable.Range(0, DetectorsRanges.Dimensions[1]))
            {
                DetectorsRanges[gradientMagnitudeIdx, gradientAngleDegreeIdx] = new DetectorRange
                {
                    GradientMagnitudeHalfRange = gradientMagnitudeRange,
                    GradientAngleHalfRange = angleRange
                };
            }
        }

        // Плотность детекторов, в зависимости от модуля градиента и угла градиента (в градусах).
        MatrixFloat_ColumnMajor detectorDensities = new MatrixFloat_ColumnMajor(DetectorsRanges.Dimensions[0], DetectorsRanges.Dimensions[1]);

        foreach (int gradientMagnitude in Enumerable.Range(0, detectorDensities.Dimensions[0]))//Enumerable.Range(constants.GeneratedMinGradientMagnitude, gradientMagnitudeRange))
        {
            foreach (int gradientAngleDegree in Enumerable.Range(0, detectorDensities.Dimensions[1]))
            {
                detectorDensities[gradientMagnitude, gradientAngleDegree] = 1.0f;
            }
        }
        
        //float activatedCount0 = gradientMagnitudeRange * DetectorsRanges[constants.GeneratedMinGradientMagnitude, 0].GradientMagnitudeRange;
        ////float detectorsCount = (detectorRanges.GradientAngleRanges[constants.GeneratedMinGradientMagnitude, 0] * 360.0f / (2 * MathF.PI)) * gradientMagnitudeRange;
        //foreach (int gradientMagnitude in Enumerable.Range(constants.GeneratedMinGradientMagnitude + gradientMagnitudeRange, constants.GeneratedMaxGradientMagnitude - constants.GeneratedMinGradientMagnitude))
        //{
        //    float gradientAngleRange1 = DetectorsRanges[gradientMagnitude - gradientMagnitudeRange + 1, 0].GradientMagnitudeRange;
        //    float activatedCount1 = 0.0f;
        //    foreach (int gm in Enumerable.Range(gradientMagnitude - gradientMagnitudeRange + 1, gradientMagnitudeRange - 1))
        //    {
        //        activatedCount1 += detectorDensities[gm, 0] * gradientAngleRange1;
        //    }                                
        //    float px = (activatedCount0 - activatedCount1) / gradientAngleRange1;
        //    foreach (int gradientAngleDegree in Enumerable.Range(0, detectorDensities.Dimensions[1]))
        //    {
        //        detectorDensities[gradientMagnitude, gradientAngleDegree] = px;
        //    }                
        //}            

        MatrixFloat_ColumnMajor detectorDensities_Accumulative = new MatrixFloat_ColumnMajor(
            DistributionHelper.GetAccumulativeDistribution(detectorDensities.Data),
            [ detectorDensities.Dimensions[0], detectorDensities.Dimensions[1] ]);            

        //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        //dataToDisplayHolder.Distribution = new ulong[360];

        foreach (int d_index in Enumerable.Range(0, Detectors.Data.Length))
        {
            int rawIndex = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative.Data);
            //var indices = detectorDensities_Accumulative.GetIndices(rawIndex);                
            var indices = (initializationRandom.Next(constants.GeneratedMaxGradientMagnitude + gradientMagnitudeRange), initializationRandom.Next(360));

            Detector detector = Detectors.Data[d_index];

            detector.AverageGradientMagnitude = indices.Item1;                
            detector.AverageGradientAngle = (float)MathHelper.DegreesToRadians(indices.Item2);                
            detector.BitIndexInHash = initializationRandom.Next(constants.HashLength);
            //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;
        }            
    }

    /// <summary>
    ///     Prepares for calculation after DeserializeOwnedData or GenerateOwnedData
    /// </summary>
    public void Prepare()
    {
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            foreach (int d_index in Enumerable.Range(0, Detectors.Data.Length))
            {
                Detector detector = Detectors.Data[d_index];

                writer.Write(detector.AverageGradientMagnitude);                    
                writer.Write(detector.AverageGradientAngle);                    
                writer.Write(detector.BitIndexInHash);
            }

            writer.WriteOwnedDataSerializable(Detectors, null);                
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            { 
                case 1:
                    foreach (int d_index in Enumerable.Range(0, Detectors.Data.Length))
                    {
                        Detector detector = Detectors.Data[d_index];

                        detector.AverageGradientMagnitude = reader.ReadSingle();                            
                        detector.AverageGradientAngle = reader.ReadSingle();                            
                        detector.BitIndexInHash = reader.ReadInt32();
                    }
                    
                    reader.ReadOwnedDataSerializable(Detectors, null);                                           
                    break;
            }
        }
    }

    #endregion        
}

public class Detector
{
    public int DI;

    public int DJ;        

    /// <summary>
    ///     
    /// </summary>
    public double CenterXPixels { get; init; }

    /// <summary>
    ///     
    /// </summary>
    public double CenterYPixels { get; init; }
    
    public float AverageGradientMagnitude;        

    /// <summary>
    ///     [-pi, pi]
    /// </summary>
    public float AverageGradientAngle;        

    public int BitIndexInHash;

    public bool Temp_IsActivated;

    public GradientInPoint Temp_GradientInPoint;

    public void CalculateIsActivated(Retina retina, DenseMatrix<GradientInPoint> gradientMatrix, IConstantsObsolete constants, Vector2 offset = default)
    {
        Temp_GradientInPoint = MathHelper.GetInterpolatedGradient(CenterXPixels - offset.X, CenterYPixels - offset.Y, gradientMatrix);

        if (Temp_GradientInPoint.Magnitude < constants.DetectorMinGradientMagnitude)
        {
            Temp_IsActivated = false;
            return;
        }
        
        DetectorRange detectorRange = retina.DetectorsRanges[(int)Temp_GradientInPoint.Magnitude, (int)Temp_GradientInPoint.Angle];
        
        bool activated = (Temp_GradientInPoint.Magnitude >= AverageGradientMagnitude - detectorRange.GradientMagnitudeHalfRange) && 
            (Temp_GradientInPoint.Magnitude < AverageGradientMagnitude + detectorRange.GradientMagnitudeHalfRange);
        if (!activated)
        {
            Temp_IsActivated = false;
            return;
        }

        // [-pi, pi)
        float gradientAngleMin = MathHelper.NormalizeAngle(AverageGradientAngle - detectorRange.GradientAngleHalfRange);
        float gradientAngleMax = MathHelper.NormalizeAngle(AverageGradientAngle + detectorRange.GradientAngleHalfRange);
        if (gradientAngleMax > gradientAngleMin + 0.01f)
            activated = (Temp_GradientInPoint.Angle >= gradientAngleMin) && (Temp_GradientInPoint.Angle < gradientAngleMax);
        else
            activated = (Temp_GradientInPoint.Angle >= gradientAngleMin) || (Temp_GradientInPoint.Angle < gradientAngleMax);
        Temp_IsActivated = activated;
    }

    public bool GetIsActivated_Obsolete(GradientInPoint[,] gradientMatrix, IConstantsObsolete constants, Vector2 offset = default)
    {
        (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(CenterXPixels - offset.X, CenterYPixels - offset.Y, gradientMatrix);

        if (magnitude < constants.DetectorMinGradientMagnitude)
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

/// <summary>
///     Диапазоны для детектора.
/// </summary>
public class DetectorRange : IOwnedDataSerializable
{
    /// <summary>
    ///     Половина диапазона модуля градиента.
    /// </summary>
    public float GradientMagnitudeHalfRange;

    /// <summary>
    ///     Половина диапазона угла градиента.
    /// </summary>
    public float GradientAngleHalfRange;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(GradientMagnitudeHalfRange);
        writer.Write(GradientAngleHalfRange);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        GradientMagnitudeHalfRange = reader.ReadSingle();
        GradientAngleHalfRange = reader.ReadSingle();
    }
}

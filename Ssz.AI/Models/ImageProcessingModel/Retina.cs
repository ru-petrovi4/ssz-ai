using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Ssz.AI.Core.Grafana;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.Utils;
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

    public Retina(IRetinaConstants constants, ILogger logger)
    {
        Logger = logger;

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

    public readonly ILogger Logger;

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
        float gmIn1 = constants.MaxGradientMagnitudeExclusive / constants.HyperColumnDefinedRadius_MiniColumns;        

        DetectorsRanges = new DenseMatrix<DetectorRange>(constants.MaxGradientMagnitudeExclusive, 360);
        
        float gradientMagnitudeRange = gmIn1 * 5;
        //float angleRange0 = MathF.Atan2(constants.K5, constants.AngleRangeDegree_LimitMagnitude / gmIn1) * 4.0f;            
        //float angleRange0 = constants.AngleRangeDegreeMin * MathF.PI / 180;
        //float angleRange1 = constants.AngleRangeDegreeMax * MathF.PI / 180;
        for (int gradientMagnitude = 0; gradientMagnitude < DetectorsRanges.Dimensions[0]; gradientMagnitude += 1)
        {   
            //if (gradientMagnitude < constants.AngleRangeDegree_LimitMagnitude)
            //    angleRange = angleRange0 + (angleRange1 - angleRange0) * (constants.AngleRangeDegree_LimitMagnitude - gradientMagnitude) / (constants.AngleRangeDegree_LimitMagnitude - constants.GeneratedMinGradientMagnitude);
            //else
            //angleRange = angleRange0;
            float fullCircle_MiniColuns = 2.0f * MathF.PI * gradientMagnitude / gmIn1;

            float gradientAngleRange_MiniColumns = 3.0f;

            float angleRange = 2.0f * MathF.PI * gradientAngleRange_MiniColumns / fullCircle_MiniColuns; // constants.K5
            
            if (Single.IsInfinity(angleRange) || angleRange > 2 * MathF.PI)
                angleRange = 2 * MathF.PI;

            for (int gradientAngleDegree = 0; gradientAngleDegree < DetectorsRanges.Dimensions[1]; gradientAngleDegree += 1)            
            {
                DetectorsRanges[gradientMagnitude, gradientAngleDegree] = new DetectorRange
                {
                    GradientMagnitudeHalfRange = gradientMagnitudeRange / 2,
                    GradientAngleHalfRange = angleRange / 2
                };
            }
        }

        // Плотность детекторов, в зависимости от модуля градиента и угла градиента (в градусах) детектора.
        MatrixFloat_ColumnMajor detectorDensities = new MatrixFloat_ColumnMajor((int)((constants.MaxGradientMagnitudeExclusive + gradientMagnitudeRange / 2.0f) / constants.GradientMagnitudeDelta), (int)(360 / constants.GradientAngleDegreeDelta));

        CalculateDetectorDensities(initializationRandom, detectorDensities, constants);        

        MatrixFloat_ColumnMajor detectorDensities_Accumulative = new MatrixFloat_ColumnMajor(
            DistributionHelper.GetAccumulativeDistribution(detectorDensities.Data),
            [detectorDensities.Dimensions[0], detectorDensities.Dimensions[1]]);        

        foreach (int d_index in Enumerable.Range(0, Detectors.Data.Length))
        {
            int dataIndex = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative.Data);
            var indices = detectorDensities_Accumulative.GetIndices(dataIndex);            

            Detector detector = Detectors.Data[d_index];

            detector.AverageGradientMagnitude = indices.I * constants.GradientMagnitudeDelta;
            detector.AverageGradientAngle = MathHelper.DegreesToRadians(indices.J * constants.GradientAngleDegreeDelta);                
            detector.BitIndexInHash = initializationRandom.Next(constants.HashLength);
            //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;
        }

        TestDetectorDensities(initializationRandom, detectorDensities_Accumulative, constants);        
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

    private void CalculateDetectorDensities(Random initializationRandom, MatrixFloat_ColumnMajor detectorDensities, IRetinaConstants constants)
    {
        DenseMatrix<Detector> testDetectors = new DenseMatrix<Detector>(detectorDensities.Dimensions[0], detectorDensities.Dimensions[1]);
        for (int dJ = 0; dJ < testDetectors.Dimensions[1]; dJ += 1)
            for(int dI = 0; dI < testDetectors.Dimensions[0]; dI += 1)
            {   
                Detector detector = new Detector();

                detector.DI = dI;
                detector.DJ = dJ;
                detector.AverageGradientMagnitude = dI * constants.GradientMagnitudeDelta;
                detector.AverageGradientAngle = MathHelper.DegreesToRadians(dJ * constants.GradientAngleDegreeDelta);
                detector.Temp_Samples = new FastList<Sample>(300);

                //detector.AverageGradientMagnitude = initializationRandom.NextSingle() * detectorDensities.Dimensions[0];
                //detector.AverageGradientAngle = MathHelper.DegreesToRadians(initializationRandom.Next(detectorDensities.Dimensions[1]));            

                testDetectors[dI, dJ] = detector;
            }

        FastList<Sample> samples = new FastList<Sample>((int)(((constants.MaxGradientMagnitudeExclusive + 1) * 361) / (constants.GradientMagnitudeDelta * constants.GradientAngleDegreeDelta)));
        for (int gradientMagnitude = (int)constants.DetectorMinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
            {
                Sample sample = new()
                {
                    gradientMagnitude = gradientMagnitude,
                    gradientAngleDegree = gradientAngleDegree,
                };
                samples.Add(sample);

                double angle = MathHelper.DegreesToRadians(gradientAngleDegree);
                GradientInPoint gradientInPoint = new GradientInPoint
                {
                    GradX = gradientMagnitude * Math.Cos(angle),
                    GradY = gradientMagnitude * Math.Sin(angle),
                    Magnitude = gradientMagnitude,
                    Angle = angle
                };

                for (int d_index = 0; d_index < testDetectors.Data.Length; d_index += 1)
                {
                    var detector = testDetectors.Data[d_index];
                    detector.CalculateIsActivated(
                        this,
                        gradientInPoint,
                        constants
                    );
                    if (detector.Temp_IsActivated)
                    {
                        sample.Detectors.Add(detector);
                        detector.Temp_Samples.Add(sample);
                    }
                }
            }
        }

        Array.Fill(detectorDensities.Data, 1.0f);

        for (; ; )
        {
            float activatedTotalAverage = 0.0f;
            for (int s_index = 0; s_index < samples.Count; s_index += 1)
            {
                var sample = samples[s_index];
                float activatedTotal = 0.0f;
                for (int d_index = 0; d_index < sample.Detectors.Count; d_index += 1)
                {
                    var detector = sample.Detectors[d_index];
                    activatedTotal += detectorDensities[detector.DI, detector.DJ];
                }
                sample.Temp_ActivatedTotal = activatedTotal;
                activatedTotalAverage += activatedTotal;
            }
            activatedTotalAverage /= samples.Count;                        

            float activatedDelta_NormAbsMax = Single.MinValue;            
            for (int s_index = 0; s_index < samples.Count; s_index += 1)
            {
                var sample = samples[s_index];                
                float activatedDelta_NormAbs = MathF.Abs(sample.Temp_ActivatedTotal - activatedTotalAverage) / activatedTotalAverage;
                if (activatedDelta_NormAbs > activatedDelta_NormAbsMax)
                    activatedDelta_NormAbsMax = activatedDelta_NormAbs;
            }

            Logger.LogInformation($"Retina.CalculateDetectorDensities, activatedDelta_NormAbsMax: {activatedDelta_NormAbsMax}");

            if (activatedDelta_NormAbsMax < 0.3)
                break;

            for (int d_index = 0; d_index < testDetectors.Data.Length; d_index += 1)
            {
                var detector = testDetectors.Data[d_index];
                if (detector.Temp_Samples.Count == 0)
                    continue;
                float detector_K = 0.0f;
                for (int s_index = 0; s_index < detector.Temp_Samples.Count; s_index += 1)
                {
                    var sample = detector.Temp_Samples[s_index];
                    detector_K += sample.Temp_ActivatedTotal / activatedTotalAverage;
                }
                detector_K /= detector.Temp_Samples.Count;

                detectorDensities[detector.DI, detector.DJ] /= detector_K;
            }

            //var randomSample = sample_AbsMax!;//samples[initializationRandom.Next(samples.Count)];
            //for (int d_index = 0; d_index < randomSample.Detectors.Count; d_index += 1)
            //{
            //    var detector = randomSample.Detectors[d_index];
            //    detectorDensities[detector.DI, detector.DJ] -= randomSample.Temp_ActivatedDelta * 0.1f;
            //}
        }
    }

    private void TestDetectorDensities(Random initializationRandom, MatrixFloat_ColumnMajor detectorDensities_Accumulative, IRetinaConstants constants)
    {
        Detector[] testDetectors = new Detector[constants.MiniColumnVisibleDetectorsCount];
        for (int d_index = 0; d_index < testDetectors.Length; d_index += 1)
        {
            int dataIndex = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative.Data);
            var indices = detectorDensities_Accumulative.GetIndices(dataIndex);

            Detector detector = new Detector();

            detector.AverageGradientMagnitude = indices.I * constants.GradientMagnitudeDelta;
            detector.AverageGradientAngle = MathHelper.DegreesToRadians(indices.J * constants.GradientAngleDegreeDelta);
            detector.BitIndexInHash = initializationRandom.Next(constants.HashLength);

            testDetectors[d_index] = detector;
        }        

        DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
        dataToDisplayHolder.DistributionXMin = 0.0f;
        dataToDisplayHolder.DistributionXMax = constants.MaxGradientMagnitudeExclusive / constants.GradientMagnitudeDelta;
        dataToDisplayHolder.Distribution = new ulong[(int)(constants.MaxGradientMagnitudeExclusive / constants.GradientMagnitudeDelta) + 1];

        for (int gradientMagnitude = (int)constants.DetectorMinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            int activatedCount = 0;
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
            {
                double angle = MathHelper.DegreesToRadians(gradientAngleDegree);
                GradientInPoint gradientInPoint = new()
                {
                    GradX = gradientMagnitude * Math.Cos(angle),
                    GradY = gradientMagnitude * Math.Sin(angle),
                    Magnitude = gradientMagnitude,
                    Angle = angle
                };

                for (int d_index = 0; d_index < testDetectors.Length; d_index += 1)
                {
                    var detector = testDetectors[d_index];
                    detector.CalculateIsActivated(
                        this,
                        gradientInPoint,
                        constants
                    );
                    if (detector.Temp_IsActivated)
                    {
                        activatedCount += 1;
                        detector.Temp_IsActivatedCount += 1;
                    }
                }
            }
            dataToDisplayHolder.Distribution[(int)(gradientMagnitude / constants.GradientMagnitudeDelta)] = (ulong)(activatedCount * constants.GradientAngleDegreeDelta / 360);
        }   
    }
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

    public int Temp_IsActivatedCount;

    public GradientInPoint Temp_GradientInPoint;

    public FastList<Sample> Temp_Samples = null!;

    public void CalculateIsActivated(Retina retina, DenseMatrix<GradientInPoint> gradientMatrix, IRetinaConstants constants, Vector2 offset = default)
    {
        CalculateIsActivated(
            retina, 
            MathHelper.GetInterpolatedGradient(CenterXPixels - offset.X, CenterYPixels - offset.Y, gradientMatrix), 
            constants);        
    }

    public void CalculateIsActivated(Retina retina, GradientInPoint gradientInPoint, IRetinaConstants constants)
    {
        Temp_GradientInPoint = gradientInPoint;

        if (gradientInPoint.Magnitude < constants.DetectorMinGradientMagnitudeInclusive || 
                gradientInPoint.Magnitude >= constants.MaxGradientMagnitudeExclusive)
        {
            Temp_IsActivated = false;
            return;
        }

        DetectorRange detectorRange = retina.DetectorsRanges[(int)gradientInPoint.Magnitude, (int)gradientInPoint.Angle];

        bool activated = (gradientInPoint.Magnitude >= AverageGradientMagnitude - detectorRange.GradientMagnitudeHalfRange) &&
            (gradientInPoint.Magnitude < AverageGradientMagnitude + detectorRange.GradientMagnitudeHalfRange);
        if (!activated)
        {
            Temp_IsActivated = false;
            return;
        }

        // [-pi, pi)
        float gradientAngleMin = MathHelper.NormalizeAngle(AverageGradientAngle - detectorRange.GradientAngleHalfRange);
        float gradientAngleMax = MathHelper.NormalizeAngle(AverageGradientAngle + detectorRange.GradientAngleHalfRange);
        if (gradientAngleMax > gradientAngleMin + 0.01f)
            activated = (gradientInPoint.Angle >= gradientAngleMin) && (gradientInPoint.Angle < gradientAngleMax);
        else
            activated = (gradientInPoint.Angle >= gradientAngleMin) || (gradientInPoint.Angle < gradientAngleMax);
        Temp_IsActivated = activated;
    }

    public bool GetIsActivated_Obsolete(GradientInPoint[,] gradientMatrix, IConstantsObsolete constants, Vector2 offset = default)
    {
        (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(CenterXPixels - offset.X, CenterYPixels - offset.Y, gradientMatrix);

        if (magnitude < constants.DetectorMinGradientMagnitudeInclusive)
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

public class Sample
{    
    public int gradientMagnitude;
    
    public int gradientAngleDegree;

    public readonly FastList<Detector> Detectors = new(300);

    public float Temp_ActivatedTotal;
}

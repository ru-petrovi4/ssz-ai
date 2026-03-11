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
        Constants = constants;
        Logger = logger;
    }

    #endregion

    #region public functions

    public readonly IRetinaConstants Constants;

    public readonly ILogger Logger;

	public FastList<GradientRange?> IdealPinwheel_GradientRanges = new();

    /// <summary>
	///     Диапазоны для детекторов в зависимости от модуля градиента и угла градиента (в градусах).
	/// </summary>
	public DenseMatrix<GradientRange?> DetectorGradientRanges = new();

    public DenseMatrix<Detector?> Detectors = new();

    public DenseMatrix<RetinaPoint> Temp_RetinaPoints = null!;

    public FastList<RetinaPoint> Temp_ToCalculateRetinaPoints = null!;

    /// <summary>
    ///     Generates model data after construction.
    /// </summary>
    public void GenerateOwnedData(Random initializationRandom, GradientDistribution gradientDistribution)
    {
        float gradientAngleRange_MiniColumns = 5.0f;
        float zeroRadius_MiniColumns = gradientAngleRange_MiniColumns / (2.0f * MathF.PI);

        int idealPinwheel_GradientMagnitudeRanges_Count = Constants.HyperColumnDefinedRadius_MiniColumns + 1;
        IdealPinwheel_GradientRanges = new FastList<GradientRange?>(idealPinwheel_GradientMagnitudeRanges_Count);
        ulong[] gradientMagnitude_AccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
        ulong samples_Total = gradientMagnitude_AccumulativeDistribution[^1];
        float inIdealPinwheelMiniColumn_Samples = samples_Total / (zeroRadius_MiniColumns + idealPinwheel_GradientMagnitudeRanges_Count - 1);        
        int gradientMagnitude_LowerInclusive = 0;        
        for (int range_Index = 0; range_Index < idealPinwheel_GradientMagnitudeRanges_Count; range_Index += 1)
        {
            ulong samples_UpperLimit = (ulong)(inIdealPinwheelMiniColumn_Samples * (zeroRadius_MiniColumns + range_Index));            

            if (samples_UpperLimit > samples_Total)
                samples_UpperLimit = samples_Total;

            int gradientMagnitude_UpperExclusive = DistributionHelper.GetIndex(samples_UpperLimit, gradientMagnitude_AccumulativeDistribution) + 1;            

            // Переделать на только Average
            IdealPinwheel_GradientRanges.Add(new GradientRange
                {
                    GradientMagnitude_LowerInclusive = gradientMagnitude_LowerInclusive,
                    GradientMagnitude_Average = (gradientMagnitude_UpperExclusive + gradientMagnitude_LowerInclusive) / 2.0f,
                    GradientMagnitude_UpperExclusive = gradientMagnitude_UpperExclusive,
                });

            gradientMagnitude_LowerInclusive = gradientMagnitude_UpperExclusive;
        }
        
        float gradientMagnitudeRange_Samples = 5.0f * inIdealPinwheelMiniColumn_Samples;

        DetectorGradientRanges = new DenseMatrix<GradientRange?>(Constants.MaxGradientMagnitudeExclusive, 360);                
        
        for (int gradientMagnitude = 0; gradientMagnitude < DetectorGradientRanges.Dimensions[0]; gradientMagnitude += 1)
        {
            float samples = gradientMagnitude_AccumulativeDistribution[gradientMagnitude];
            float idealPinwheel_MiniColumns = samples / inIdealPinwheelMiniColumn_Samples;
            
            float fullCircle_MiniColuns = 2.0f * MathF.PI * idealPinwheel_MiniColumns;            
            float angleRange = 2.0f * MathF.PI * gradientAngleRange_MiniColumns / fullCircle_MiniColuns;
            
            if (Single.IsNaN(angleRange) || Single.IsInfinity(angleRange) || angleRange > 2 * MathF.PI)
                angleRange = 2 * MathF.PI;

            long samples_Lower = (long)(samples - gradientMagnitudeRange_Samples / 2.0f);
            if (samples_Lower < 0)
                samples_Lower = 0;
            long samples_Upper = (long)(samples + gradientMagnitudeRange_Samples / 2.0f);
            if (samples_Upper > (long)samples_Total)
                samples_Upper = (long)samples_Total;               

            for (int gradientAngleDegree = 0; gradientAngleDegree < DetectorGradientRanges.Dimensions[1]; gradientAngleDegree += 1)            
            {
                float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);
                DetectorGradientRanges[gradientMagnitude, gradientAngleDegree] = new GradientRange
                {
                    GradientMagnitude_LowerInclusive = DistributionHelper.GetIndex((ulong)samples_Lower, gradientMagnitude_AccumulativeDistribution),
                    GradientMagnitude_Average = gradientMagnitude,
                    GradientMagnitude_UpperExclusive = DistributionHelper.GetIndex((ulong)samples_Upper, gradientMagnitude_AccumulativeDistribution) + 1,
                    GradientAngle_LowerInclusive = MathHelper.NormalizeAngle(gradientAngle - angleRange / 2.0f),
                    GradientAngle_Average = gradientAngle,
                    GradientAngle_UpperExclusive = MathHelper.NormalizeAngle(gradientAngle + angleRange / 2.0f)
                };
            }
        }

        MatrixFloat_ColumnMajor detectorDensities_ShortIndexed = CalculateDetectorDensities(initializationRandom, Constants);
        
        Detectors = new DenseMatrix<Detector?>(
            (int)(Constants.RetinaImagePixelSize.Width / Constants.RetinaDetectorsDeltaPixels),
            (int)(Constants.RetinaImagePixelSize.Height / Constants.RetinaDetectorsDeltaPixels));        

        MatrixFloat_ColumnMajor detectorDensities_Accumulative_ShortIndexed = new MatrixFloat_ColumnMajor(
            DistributionHelper.GetAccumulativeDistribution(detectorDensities_ShortIndexed.Data),
            [detectorDensities_ShortIndexed.Dimensions[0], detectorDensities_ShortIndexed.Dimensions[1]]);

        foreach (int dJ in Enumerable.Range(0, Detectors.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, Detectors.Dimensions[0]))
            {
                int dataIndex = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative_ShortIndexed.Data);
                var indices_ShortIndexed = detectorDensities_Accumulative_ShortIndexed.GetIndices(dataIndex);

                Detector detector = new()
                {
                    Retina = this,
                    DI = dI,
                    DJ = dJ,
                    CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                    CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                };
                detector.GradientMagnitude_Average = indices_ShortIndexed.I * Constants.GradientMagnitudeDelta;
                detector.GradientAngle_Average = MathHelper.DegreesToRadians(indices_ShortIndexed.J * Constants.GradientAngleDegreeDelta);                            
                detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

                Detectors[dI, dJ] = detector;
            }

        TestDetectorDensities(initializationRandom, detectorDensities_Accumulative_ShortIndexed, Constants);        
    }    

    /// <summary>
    ///     Prepares for calculation after DeserializeOwnedData or GenerateOwnedData
    /// </summary>
    public void Prepare()
    {
        Temp_RetinaPoints = new DenseMatrix<RetinaPoint>((int)(Constants.RetinaImagePixelSize.Width / Constants.RetinaPointDeltaPixels), (int)(Constants.RetinaImagePixelSize.Height / Constants.RetinaPointDeltaPixels));
        Temp_RetinaPoints.CreateElementInstances((int x, int y) => new RetinaPoint()
        {
            CenterXPixels = x * Constants.RetinaPointDeltaPixels,
            CenterYPixels = y * Constants.RetinaPointDeltaPixels
        });

        float detectorFieldOfViewRadiusPixels = Constants.DetectorFieldOfViewRadiusPixels;
        foreach (int dJ in Enumerable.Range(0, Detectors.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, Detectors.Dimensions[0]))
            {
                Detector detector = Detectors[dI, dJ]!;
                detector.Temp_RetinaPoints = new FastList<RetinaPoint>((int)(MathF.PI * (1 + detectorFieldOfViewRadiusPixels / Constants.RetinaPointDeltaPixels) * (1 + detectorFieldOfViewRadiusPixels / Constants.RetinaPointDeltaPixels)));

                for (int rpJ = (int)((detector.CenterYPixels - detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels); rpJ < (int)((detector.CenterYPixels + detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels) && rpJ < Temp_RetinaPoints.Dimensions[1]; rpJ += 1)
                    for (int rpI = (int)((detector.CenterXPixels - detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels); rpI < (int)((detector.CenterXPixels + detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels) && rpI < Temp_RetinaPoints.Dimensions[0]; rpI += 1)
                    {
                        if (rpI < 0 || rpJ < 0)
                            continue;

                        RetinaPoint retinaPoint = Temp_RetinaPoints[rpI, rpJ]!;
                        double rPixels = Math.Sqrt((detector.CenterXPixels - retinaPoint.CenterXPixels) * (detector.CenterXPixels - retinaPoint.CenterXPixels) + (detector.CenterYPixels - retinaPoint.CenterYPixels) * (detector.CenterYPixels - retinaPoint.CenterYPixels));
                        if (rPixels < detectorFieldOfViewRadiusPixels)
                            detector.Temp_RetinaPoints.Add(retinaPoint);
                    }
            }

        Temp_ToCalculateRetinaPoints = new FastList<RetinaPoint>(Temp_RetinaPoints.Data);
    }

    public void CalculateRetinaPoints(DenseMatrix<GradientInPoint> eye_GradientMatrix)
    {        
        for (int rp_Index = 0; rp_Index < Temp_ToCalculateRetinaPoints.Count; rp_Index += 1)
        {
            RetinaPoint retinaPoint = Temp_ToCalculateRetinaPoints[rp_Index];
            retinaPoint.GradientInPoint = MathHelper.GetInterpolatedGradient(
                retinaPoint.CenterXPixels,
                retinaPoint.CenterYPixels,
                eye_GradientMatrix);
        }
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {			
			Helpers.SerializationHelper.SerializeOwnedData_FastList(IdealPinwheel_GradientRanges, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(DetectorGradientRanges, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(Detectors, writer, context);            
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            { 
                case 1:
					IdealPinwheel_GradientRanges = Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, (int index) => new GradientRange());
                    DetectorGradientRanges = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new GradientRange());
                    Detectors = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new Detector
                    {
                        Retina = this,
                        DI = dI,
                        DJ = dJ,
                        CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                        CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                    });                    
                    break;
            }
        }
    }

    #endregion

    private MatrixFloat_ColumnMajor CalculateDetectorDensities(Random initializationRandom, IRetinaConstants constants)
    {   
        // Плотность детекторов, в зависимости от модуля градиента и угла градиента (в градусах) детектора.
        MatrixFloat_ColumnMajor detectorDensities_ShortIndexed = new MatrixFloat_ColumnMajor(
            (int)(constants.MaxGradientMagnitudeExclusive / Constants.GradientMagnitudeDelta), //(int)(Constants.MaxGradientMagnitudeExclusive / Constants.GradientMagnitudeDelta)
            (int)(360 / Constants.GradientAngleDegreeDelta));

        DenseMatrix<Detector> testDetectors_ShortIndexed = new DenseMatrix<Detector>(detectorDensities_ShortIndexed.Dimensions[0], detectorDensities_ShortIndexed.Dimensions[1]);
        for (int dJ_ShortIndexed = 0; dJ_ShortIndexed < detectorDensities_ShortIndexed.Dimensions[1]; dJ_ShortIndexed += 1)
            for(int dI_ShortIndexed = 0; dI_ShortIndexed < detectorDensities_ShortIndexed.Dimensions[0]; dI_ShortIndexed += 1)
            {
                Detector detector = new Detector
                {
                    Retina = this,
                    DI = dI_ShortIndexed,
                    DJ = dJ_ShortIndexed,
                };
                detector.GradientMagnitude_Average = dI_ShortIndexed * Constants.GradientMagnitudeDelta;
                detector.GradientAngle_Average = MathHelper.DegreesToRadians(dJ_ShortIndexed * Constants.GradientAngleDegreeDelta);                
                detector.Temp_GradientSamples = new FastList<GradientSample>(300); 

                testDetectors_ShortIndexed[dI_ShortIndexed, dJ_ShortIndexed] = detector;
            }

        FastList<GradientSample> gradientSamples = new FastList<GradientSample>((int)(constants.MaxGradientMagnitudeExclusive * 360 / (constants.GradientMagnitudeDelta * constants.GradientAngleDegreeDelta)));
        for (int gradientMagnitude = (int)constants.MinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
            {
                GradientSample gradientSample = new()
                {
                    GradientMagnitude = gradientMagnitude,
                    GradientAngleDegree = gradientAngleDegree,
                };
                gradientSamples.Add(gradientSample);

                double angle = MathHelper.DegreesToRadians(gradientAngleDegree);
                GradientInPoint gradientInPoint = new GradientInPoint
                {
                    GradX = gradientMagnitude * Math.Cos(angle),
                    GradY = gradientMagnitude * Math.Sin(angle),
                    Magnitude = gradientMagnitude,
                    Angle = angle
                };

                for (int d_index = 0; d_index < testDetectors_ShortIndexed.Data.Length; d_index += 1)
                {
                    var detector = testDetectors_ShortIndexed.Data[d_index];                    
                    if (detector.CalculateIsActivated(gradientInPoint))
                    {
                        gradientSample.Detectors.Add(detector);
                        detector.Temp_GradientSamples.Add(gradientSample);
                    }
                }
            }
        }

        Array.Fill(detectorDensities_ShortIndexed.Data, 1.0f);

        float prev_ActivatedDelta_NormAbsMax = Single.MaxValue;
        for (; ; )
        {
            float activatedTotalAverage = 0.0f;
            for (int s_index = 0; s_index < gradientSamples.Count; s_index += 1)
            {
                var gradientSample = gradientSamples[s_index];
                float activatedTotal = 0.0f;
                for (int d_index = 0; d_index < gradientSample.Detectors.Count; d_index += 1)
                {
                    var detector = gradientSample.Detectors[d_index];
                    activatedTotal += detectorDensities_ShortIndexed[detector.DI, detector.DJ];
                }
                gradientSample.Temp_ActivatedTotal = activatedTotal;
                activatedTotalAverage += activatedTotal;
            }
            activatedTotalAverage /= gradientSamples.Count;                        

            float activatedDelta_NormAbsMax = Single.MinValue;            
            for (int s_index = 0; s_index < gradientSamples.Count; s_index += 1)
            {
                var gradientSample = gradientSamples[s_index];                
                float activatedDelta_NormAbs = MathF.Abs(gradientSample.Temp_ActivatedTotal - activatedTotalAverage) / activatedTotalAverage;
                if (activatedDelta_NormAbs > activatedDelta_NormAbsMax)
                    activatedDelta_NormAbsMax = activatedDelta_NormAbs;
            }

            Logger.LogInformation($"Retina.CalculateDetectorDensities, activatedDelta_NormAbsMax: {activatedDelta_NormAbsMax}");

            if (activatedDelta_NormAbsMax > prev_ActivatedDelta_NormAbsMax - 0.001f)
                break;

            prev_ActivatedDelta_NormAbsMax = activatedDelta_NormAbsMax;

            for (int d_index = 0; d_index < testDetectors_ShortIndexed.Data.Length; d_index += 1)
            {
                var detector = testDetectors_ShortIndexed.Data[d_index];
                if (detector.Temp_GradientSamples.Count == 0)
                {
                    detectorDensities_ShortIndexed[detector.DI, detector.DJ] = 0.0f;
                    continue;
                }
                float detector_K = 0.0f;
                for (int s_index = 0; s_index < detector.Temp_GradientSamples.Count; s_index += 1)
                {
                    var gradientSample = detector.Temp_GradientSamples[s_index];
                    detector_K += gradientSample.Temp_ActivatedTotal / activatedTotalAverage;
                }
                detector_K /= detector.Temp_GradientSamples.Count;

                detectorDensities_ShortIndexed[detector.DI, detector.DJ] /= detector_K;
            }

            //var randomSample = sample_AbsMax!;//samples[initializationRandom.Next(samples.Count)];
            //for (int d_index = 0; d_index < randomSample.Detectors.Count; d_index += 1)
            //{
            //    var detector = randomSample.Detectors[d_index];
            //    detectorDensities[detector.DI, detector.DJ] -= randomSample.Temp_ActivatedDelta * 0.1f;
            //}
        }

        return detectorDensities_ShortIndexed;
    }

    private void TestDetectorDensities(Random initializationRandom, MatrixFloat_ColumnMajor detectorDensities_Accumulative_ShortIndexed, IRetinaConstants constants)
    {
        Detector[] testDetectors = new Detector[constants.MiniColumnVisibleDetectorsCount];
        for (int d_index = 0; d_index < testDetectors.Length; d_index += 1)
        {
            int dataIndex = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative_ShortIndexed.Data);
            var indices_ShortIndexed = detectorDensities_Accumulative_ShortIndexed.GetIndices(dataIndex);

            Detector detector = new Detector()
            {
                Retina = this,
            };
            detector.GradientMagnitude_Average = indices_ShortIndexed.I * Constants.GradientMagnitudeDelta;
            detector.GradientAngle_Average = MathHelper.DegreesToRadians(indices_ShortIndexed.J * Constants.GradientAngleDegreeDelta);            
            detector.BitIndexInHash = initializationRandom.Next(constants.HashLength);

            testDetectors[d_index] = detector;
        }        

        DataToDisplayHolder dataToDisplayHolder = DataToDisplayHolder.Instance;
        dataToDisplayHolder.DistributionXMin = 0.0f;
        dataToDisplayHolder.DistributionXMax = constants.MaxGradientMagnitudeExclusive / constants.GradientMagnitudeDelta;
        dataToDisplayHolder.Distribution = new ulong[(int)(constants.MaxGradientMagnitudeExclusive / constants.GradientMagnitudeDelta) + 1];

        for (int gradientMagnitude = (int)constants.MinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
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
                    if (detector.CalculateIsActivated(gradientInPoint))
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

public class Detector : IOwnedDataSerializable
{
    public Retina Retina = null!;

    public int DI;

    public int DJ;

    public double CenterXPixels;

    public double CenterYPixels;

    /// <summary>
    ///     [0, Constants.MaxGradientMagnitudeInclusive)
    /// </summary>
    public float GradientMagnitude_Average;    

    /// <summary>
    ///     [-pi, pi)
    /// </summary>
    public float GradientAngle_Average;

    public int BitIndexInHash;

    public bool Temp_IsActivated;

    public int Temp_IsActivatedCount;    

    public FastList<GradientSample> Temp_GradientSamples = null!;

    public FastList<RetinaPoint> Temp_RetinaPoints = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(GradientMagnitude_Average);
        writer.Write(GradientAngle_Average);
        writer.Write(BitIndexInHash);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        GradientMagnitude_Average = reader.ReadSingle();
        GradientAngle_Average = reader.ReadSingle();
        BitIndexInHash = reader.ReadInt32();
    }

    /// <summary>
    ///     Precondition: !!! Gradient in Temp_RetinaPoints must be calculated !!!
    /// </summary>
    public bool CalculateIsActivated()
    {
        for (int rp_Index = 0; rp_Index < Temp_RetinaPoints.Count; rp_Index += 1)
        {
            bool activated = CalculateIsActivated(Temp_RetinaPoints[rp_Index].GradientInPoint);
            if (activated)
                return true;
        }
        return false;
    }

    public bool CalculateIsActivated(GradientInPoint gradientInPoint)
    {
        if (gradientInPoint.Magnitude < Retina.Constants.MinGradientMagnitudeInclusive || 
                gradientInPoint.Magnitude >= Retina.Constants.MaxGradientMagnitudeExclusive)
            return false;

        GradientRange detectorGradientRange = Retina.DetectorGradientRanges[(int)gradientInPoint.Magnitude, (int)MathHelper.RadiansToDegrees((float)gradientInPoint.Angle)]!;

        bool activated = GradientMagnitude_Average >= detectorGradientRange.GradientMagnitude_LowerInclusive &&
            GradientMagnitude_Average < detectorGradientRange.GradientMagnitude_UpperExclusive;
        if (!activated)
            return false;

        // [-pi, pi)
        float gradientAngleMinInclusive = detectorGradientRange.GradientAngle_LowerInclusive;
        float gradientAngleMaxExclusive = detectorGradientRange.GradientAngle_UpperExclusive;
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

/// <summary>
///     Диапазон модуля и угла градиента.
/// </summary>
public class GradientRange : IOwnedDataSerializable
{    
    public float GradientMagnitude_LowerInclusive;

    /// <summary>
    ///     [0, Constants.MaxGradientMagnitudeInclusive)
    /// </summary>
    public float GradientMagnitude_Average;

    public float GradientMagnitude_UpperExclusive;

    /// <summary>
    ///     [-pi, pi)
    /// </summary>
    public float GradientAngle_LowerInclusive;

    /// <summary>
    ///     [-pi, pi)
    /// </summary>
    public float GradientAngle_Average;

    /// <summary>
    ///     [-pi, pi)
    /// </summary>
    public float GradientAngle_UpperExclusive;    

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(GradientMagnitude_LowerInclusive);
        writer.Write(GradientMagnitude_Average);
        writer.Write(GradientMagnitude_UpperExclusive);
        writer.Write(GradientAngle_LowerInclusive);
        writer.Write(GradientAngle_Average);
        writer.Write(GradientAngle_UpperExclusive);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        GradientMagnitude_LowerInclusive = reader.ReadSingle();
        GradientMagnitude_Average = reader.ReadSingle();
        GradientMagnitude_UpperExclusive = reader.ReadSingle();
        GradientAngle_LowerInclusive = reader.ReadSingle();
        GradientAngle_Average = reader.ReadSingle();
        GradientAngle_UpperExclusive = reader.ReadSingle();
    }
}

public class GradientSample
{
    public int GradientMagnitude;

    public int GradientAngleDegree;

    public readonly FastList<Detector> Detectors = new(300);

    public float Temp_ActivatedTotal;
}

public class RetinaPoint
{
    public GradientInPoint GradientInPoint;

    public double CenterXPixels { get; init; }

    public double CenterYPixels { get; init; }
}
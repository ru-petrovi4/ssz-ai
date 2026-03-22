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

    public ulong[] GradientMagnitude_AccumulativeDistribution = null!;

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
        GradientMagnitude_AccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
        ulong samples_Total = GradientMagnitude_AccumulativeDistribution[^1];
        float inIdealPinwheelMiniColumn_Samples = (float)samples_Total / Constants.HyperColumnDefinedRadius_MiniColumns;                
        
        float gradientMagnitudeRange_Samples = Constants.DetectorRange_MiniColumns * inIdealPinwheelMiniColumn_Samples; // 5.0 уже плохо собирается
        float gradientAngleRange_MiniColumns = Constants.DetectorRange_MiniColumns; // 5.0 уже плохо собирается

        DetectorGradientRanges = new DenseMatrix<GradientRange?>(Constants.MaxGradientMagnitudeExclusive, 360);                
        
        for (int gradientMagnitude = 0; gradientMagnitude < DetectorGradientRanges.Dimensions[0]; gradientMagnitude += 1)
        {
            float samples = GradientMagnitude_AccumulativeDistribution[gradientMagnitude];
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

            GradientRange? gradientRange = null;
            for (int gradientAngleDegree = 0; gradientAngleDegree < DetectorGradientRanges.Dimensions[1]; gradientAngleDegree += 1)            
            {
                float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);
                gradientRange = new GradientRange
                {
                    GradientMagnitude_LowerInclusive = DistributionHelper.GetIndex((ulong)samples_Lower, GradientMagnitude_AccumulativeDistribution),
                    GradientMagnitude_Average = gradientMagnitude,
                    GradientMagnitude_UpperExclusive = DistributionHelper.GetIndex((ulong)samples_Upper, GradientMagnitude_AccumulativeDistribution) + 1,
                    GradientAngle_LowerInclusive = MathHelper.NormalizeAngle(gradientAngle - angleRange / 2.0f),
                    GradientAngle_Average = gradientAngle,
                    GradientAngle_UpperExclusive = MathHelper.NormalizeAngle(gradientAngle + angleRange / 2.0f),
                    GradientMagnitude_Average_IdealPinwheel_MiniColumns = idealPinwheel_MiniColumns
                };
                float gradientMagnitude_UpperHalfRange = gradientRange.GradientMagnitude_UpperExclusive - gradientRange.GradientMagnitude_Average;
                float gradientMagnitude_LowerHalfRange = gradientRange.GradientMagnitude_Average - gradientRange.GradientMagnitude_LowerInclusive;
                if (samples_Upper == (long)samples_Total && gradientMagnitude_UpperHalfRange < gradientMagnitude_LowerHalfRange)
                    gradientRange.GradientMagnitude_UpperExclusive = gradientRange.GradientMagnitude_Average + gradientMagnitude_LowerHalfRange;
                else if (samples_Lower == 0 && gradientMagnitude_LowerHalfRange < gradientMagnitude_UpperHalfRange)
                    gradientRange.GradientMagnitude_LowerInclusive = gradientRange.GradientMagnitude_Average - gradientMagnitude_UpperHalfRange;
                DetectorGradientRanges[gradientMagnitude, gradientAngleDegree] = gradientRange;
            }
        }

        FastList<Detector> templateDetectors = CalculateTemplateDetectors(initializationRandom, Constants);
        
        Detectors = new DenseMatrix<Detector?>(
            (int)(Constants.RetinaImagePixelSize.Width / Constants.RetinaDetectorsDeltaPixels),
            (int)(Constants.RetinaImagePixelSize.Height / Constants.RetinaDetectorsDeltaPixels));        

        float[] detectorDensities_Accumulative = DistributionHelper.GetAccumulativeDistribution(templateDetectors.Select(d => d.Temp_Density).ToArray());

        foreach (int dJ in Enumerable.Range(0, Detectors.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, Detectors.Dimensions[0]))
            {
                int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
                Detector templateDetector = templateDetectors[index];

                Detector detector = new()
                {
                    Retina = this,
                    DI = dI,
                    DJ = dJ,
                    CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                    CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                };
                detector.GradientMagnitude_Average = templateDetector.GradientMagnitude_Average;
                detector.GradientAngle_Average = templateDetector.GradientAngle_Average;                            
                detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

                Detectors[dI, dJ] = detector;
            }

        TestDetectorDensities(initializationRandom, templateDetectors, detectorDensities_Accumulative);        
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

    public DenseMatrix<GradientInPoint> Get_Ideal_Eye_GradientMatrix(
        float testGradientAngleDegrees,
        float testGradientMagnitude,
        float testGradientWidthRelative,
        float testGradientPositionRelative)
    {
        return SobelOperator.ApplySobel(
            testGradientAngleDegrees,
            testGradientMagnitude,
            testGradientWidthRelative,
            testGradientPositionRelative,
            Constants.RetinaImagePixelSize.Width, 
            Constants.RetinaImagePixelSize.Height);
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
            writer.WriteArrayOfUInt64(GradientMagnitude_AccumulativeDistribution);
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
                    GradientMagnitude_AccumulativeDistribution = reader.ReadArrayOfUInt64();
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

    private FastList<Detector> CalculateTemplateDetectors(Random initializationRandom, IRetinaConstants constants)
    {           
        FastList<Detector> templateDetectors = new FastList<Detector>(10000);
        FastList<Detector> optimal_TemplateDetectors = new FastList<Detector>(10000);
        for (int gradientMagnitude = (int)DetectorGradientRanges[0, 0]!.GradientMagnitude_LowerInclusive; gradientMagnitude < DetectorGradientRanges[DetectorGradientRanges.Dimensions[0] - 1, 0]!.GradientMagnitude_UpperExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
            {                
                float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

                Detector detector = new Detector
                {
                    Retina = this,                    
                };
                detector.GradientMagnitude_Average = gradientMagnitude;
                detector.GradientAngle_Average = gradientAngle;
                detector.Temp_GradientSamples = new FastList<GradientSample>(300);
                detector.Temp_Density = 1.0f;
                templateDetectors.Add(detector);

                detector = new Detector
                {
                    Retina = this,
                };
                detector.GradientMagnitude_Average = gradientMagnitude;
                detector.GradientAngle_Average = gradientAngle;
                optimal_TemplateDetectors.Add(detector);
            }
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

                double gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);
                GradientInPoint gradientInPoint = new GradientInPoint
                {
                    GradX = gradientMagnitude * Math.Cos(gradientAngle),
                    GradY = gradientMagnitude * Math.Sin(gradientAngle),
                    Magnitude = gradientMagnitude,
                    Angle = gradientAngle
                };

                for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
                {
                    var templateDetector = templateDetectors[d_Index];                    
                    if (templateDetector.CalculateIsActivated(gradientInPoint))
                    {
                        gradientSample.Detectors.Add(templateDetector);
                        templateDetector.Temp_GradientSamples.Add(gradientSample);
                    }
                }

                if (gradientSample.Detectors.Count > 0)
                    gradientSamples.Add(gradientSample);
                else
                    throw new InvalidOperationException();
            }
        }

        float min_ActivatedDelta_NormAbsMax = Single.MaxValue;
        for (; ; )
        {
            float activatedTotalAverage = 0.0f;
            for (int s_Index = 0; s_Index < gradientSamples.Count; s_Index += 1)
            {
                var gradientSample = gradientSamples[s_Index];
                float activatedTotal = 0.0f;
                for (int d_Index = 0; d_Index < gradientSample.Detectors.Count; d_Index += 1)
                {
                    var templateDetector = gradientSample.Detectors[d_Index];
                    activatedTotal += templateDetector.Temp_Density;
                }
                gradientSample.Temp_ActivatedTotal = activatedTotal;
                activatedTotalAverage += activatedTotal;
            }
            activatedTotalAverage /= gradientSamples.Count;                        

            float activatedDelta_NormAbsMax = Single.MinValue;            
            for (int s_Index = 0; s_Index < gradientSamples.Count; s_Index += 1)
            {
                var gradientSample = gradientSamples[s_Index];                
                float activatedDelta_NormAbs = MathF.Abs(gradientSample.Temp_ActivatedTotal - activatedTotalAverage) / activatedTotalAverage;
                if (activatedDelta_NormAbs > activatedDelta_NormAbsMax)
                    activatedDelta_NormAbsMax = activatedDelta_NormAbs;
            }

            Logger.LogInformation($"Retina.CalculateDetectorDensities, activatedDelta_NormAbsMax: {activatedDelta_NormAbsMax}");

            if (activatedDelta_NormAbsMax > min_ActivatedDelta_NormAbsMax - 0.00001f)
                break;

            if (activatedDelta_NormAbsMax < min_ActivatedDelta_NormAbsMax)
            {
                min_ActivatedDelta_NormAbsMax = activatedDelta_NormAbsMax;
                for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
                {
                    optimal_TemplateDetectors[d_Index].Temp_Density = templateDetectors[d_Index].Temp_Density;
                }
            }

            float densityAverage = 0.0f;
            for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
            {
                var templateDetector = templateDetectors[d_Index];
                if (templateDetector.Temp_GradientSamples.Count == 0)
                {
                    templateDetector.Temp_Density = 0.0f;
                    continue;
                }
                float detector_K = 0.0f;
                for (int s_Index = 0; s_Index < templateDetector.Temp_GradientSamples.Count; s_Index += 1)
                {
                    var gradientSample = templateDetector.Temp_GradientSamples[s_Index];
                    detector_K += gradientSample.Temp_ActivatedTotal / activatedTotalAverage;
                }
                detector_K /= templateDetector.Temp_GradientSamples.Count;

                if (detector_K == 0.0f)
                    throw new InvalidOperationException();

                templateDetector.Temp_Density /= detector_K;
                densityAverage += templateDetector.Temp_Density;
            }
            densityAverage /= templateDetectors.Count;

            for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
            {
                var templateDetector = templateDetectors[d_Index];
                templateDetector.Temp_Density /= densityAverage;
            }
        }

        return optimal_TemplateDetectors;
    }

    private void TestDetectorDensities(Random initializationRandom, FastList<Detector> templateDetectors, float[] detectorDensities_Accumulative)
    {
        Detector[] testDetectors = new Detector[Constants.MiniColumnVisibleDetectorsCount];
        for (int d_Index = 0; d_Index < testDetectors.Length; d_Index += 1)
        {
            int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
            Detector templateDetector = templateDetectors[index];

            Detector detector = new Detector()
            {
                Retina = this,
            };
            detector.GradientMagnitude_Average = templateDetector.GradientMagnitude_Average;
            detector.GradientAngle_Average = templateDetector.GradientAngle_Average;
            detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);

            testDetectors[d_Index] = detector;
        }        

        DataToDisplayHolder dataToDisplayHolder = DataToDisplayHolder.Instance;
        dataToDisplayHolder.DistributionXMin = 0.0f;
        dataToDisplayHolder.DistributionXMax = Constants.MaxGradientMagnitudeExclusive / Constants.GradientMagnitudeDelta;
        dataToDisplayHolder.Distribution = new ulong[(int)(Constants.MaxGradientMagnitudeExclusive / Constants.GradientMagnitudeDelta) + 1];

        for (int gradientMagnitude = (int)Constants.MinGradientMagnitudeInclusive; gradientMagnitude < Constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)Constants.GradientMagnitudeDelta)
        {
            int activatedCount = 0;
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)Constants.GradientAngleDegreeDelta)
            {
                double angle = MathHelper.DegreesToRadians(gradientAngleDegree);
                GradientInPoint gradientInPoint = new()
                {
                    GradX = gradientMagnitude * Math.Cos(angle),
                    GradY = gradientMagnitude * Math.Sin(angle),
                    Magnitude = gradientMagnitude,
                    Angle = angle
                };

                for (int d_Index = 0; d_Index < testDetectors.Length; d_Index += 1)
                {
                    var detector = testDetectors[d_Index];                    
                    if (detector.CalculateIsActivated(gradientInPoint))
                    {
                        activatedCount += 1;
                        detector.Temp_IsActivatedCount += 1;
                    }
                }
            }
            dataToDisplayHolder.Distribution[(int)(gradientMagnitude / Constants.GradientMagnitudeDelta)] = (ulong)(activatedCount * Constants.GradientAngleDegreeDelta / 360);
        }

        Logger.LogInformation($"TestDetectorDensities: " + String.Join(", ", dataToDisplayHolder.Distribution.Select((v, i) => $"{v}")));
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

    public float Temp_Density;

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

    /// <summary>
    ///     For future use
    /// </summary>
    public float GradientMagnitude_Average_IdealPinwheel_MiniColumns;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(GradientMagnitude_LowerInclusive);
        writer.Write(GradientMagnitude_Average);
        writer.Write(GradientMagnitude_UpperExclusive);
        writer.Write(GradientAngle_LowerInclusive);
        writer.Write(GradientAngle_Average);
        writer.Write(GradientAngle_UpperExclusive);
        writer.Write(GradientMagnitude_Average_IdealPinwheel_MiniColumns);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        GradientMagnitude_LowerInclusive = reader.ReadSingle();
        GradientMagnitude_Average = reader.ReadSingle();
        GradientMagnitude_UpperExclusive = reader.ReadSingle();
        GradientAngle_LowerInclusive = reader.ReadSingle();
        GradientAngle_Average = reader.ReadSingle();
        GradientAngle_UpperExclusive = reader.ReadSingle();
        GradientMagnitude_Average_IdealPinwheel_MiniColumns = reader.ReadSingle();
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


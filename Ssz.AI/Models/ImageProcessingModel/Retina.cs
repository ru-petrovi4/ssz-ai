using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Ssz.AI.Core.Grafana;
using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.AI.Models.Primitives;
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
    ///     Диапазоны для детекторов в зависимости от модуля градиента (в градусах).
    /// </summary>
    public DenseMatrix<DetectorValueRange?> GradientMagnitude_DetectorValueRanges = new();

    public DenseMatrix<SimpleDetector?> GradientMagnitude_Detectors = new();    

    /// <summary>
    ///     Диапазоны для детекторов в зависимости от угла градиента (в градусах).
    /// </summary>
    public DenseMatrix<DetectorValueRange?> GradientAngle_DetectorValueRanges = new();

    public DenseMatrix<SimpleDetector?> GradientAngle_Detectors = new();

    public DenseMatrix<GradientComplexDetector?> GradientComplex_Detectors = new();

    public DenseMatrix<float> GradientMagnitude_Average_IdealPinwheel_MiniColumns = new();

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

        GradientMagnitude_Average_IdealPinwheel_MiniColumns = new DenseMatrix<float>(Constants.MaxGradientMagnitudeExclusive, 360);
        for (int gradientMagnitude = 0; gradientMagnitude < GradientMagnitude_DetectorValueRanges.Dimensions[0]; gradientMagnitude += 1)
        {
            float samples = GradientMagnitude_AccumulativeDistribution[gradientMagnitude];
            float idealPinwheel_MiniColumns = samples / inIdealPinwheelMiniColumn_Samples;

            for (int gradientAngleDegree = 0; gradientAngleDegree < GradientMagnitude_DetectorValueRanges.Dimensions[1]; gradientAngleDegree += 1)
            { 
                GradientMagnitude_Average_IdealPinwheel_MiniColumns[gradientMagnitude, gradientAngleDegree] = idealPinwheel_MiniColumns;
            }
        }

        GradientMagnitude_DetectorValueRanges = new DenseMatrix<DetectorValueRange?>(Constants.MaxGradientMagnitudeExclusive, 360);
        for (int gradientMagnitude = 0; gradientMagnitude < GradientMagnitude_DetectorValueRanges.Dimensions[0]; gradientMagnitude += 1)
        {
            float samples = GradientMagnitude_AccumulativeDistribution[gradientMagnitude];            

            long samples_Lower = (long)(samples - gradientMagnitudeRange_Samples / 2.0f);
            if (samples_Lower < 0)
                samples_Lower = 0;
            long samples_Upper = (long)(samples + gradientMagnitudeRange_Samples / 2.0f);
            if (samples_Upper > (long)samples_Total)
                samples_Upper = (long)samples_Total;
            
            for (int gradientAngleDegree = 0; gradientAngleDegree < GradientMagnitude_DetectorValueRanges.Dimensions[1]; gradientAngleDegree += 1)            
            {                
                DetectorValueRange detectorValueRange = new()
                {
                    LowerInclusive = DistributionHelper.GetIndex((ulong)samples_Lower, GradientMagnitude_AccumulativeDistribution),
                    Average = gradientMagnitude,
                    UpperExclusive = DistributionHelper.GetIndex((ulong)samples_Upper, GradientMagnitude_AccumulativeDistribution) + 1,
                };
                float gradientMagnitude_UpperHalfRange = detectorValueRange.UpperExclusive - detectorValueRange.Average;
                float gradientMagnitude_LowerHalfRange = detectorValueRange.Average - detectorValueRange.LowerInclusive;
                if (samples_Upper == (long)samples_Total && gradientMagnitude_UpperHalfRange < gradientMagnitude_LowerHalfRange)
                    detectorValueRange.UpperExclusive = detectorValueRange.Average + gradientMagnitude_LowerHalfRange;
                else if (samples_Lower == 0 && gradientMagnitude_LowerHalfRange < gradientMagnitude_UpperHalfRange)
                    detectorValueRange.LowerInclusive = detectorValueRange.Average - gradientMagnitude_UpperHalfRange;
                GradientMagnitude_DetectorValueRanges[gradientMagnitude, gradientAngleDegree] = detectorValueRange;
            }
        }

        GradientAngle_DetectorValueRanges = new DenseMatrix<DetectorValueRange?>(Constants.MaxGradientMagnitudeExclusive, 360);
        for (int gradientMagnitude = 0; gradientMagnitude < GradientAngle_DetectorValueRanges.Dimensions[0]; gradientMagnitude += 1)
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

            for (int gradientAngleDegree = 0; gradientAngleDegree < GradientMagnitude_DetectorValueRanges.Dimensions[1]; gradientAngleDegree += 1)
            {
                float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);
                DetectorValueRange detectorValueRange = new()
                {
                    LowerInclusive = MathHelper.NormalizeAngle(gradientAngle - angleRange / 2.0f),
                    Average = gradientAngle,
                    UpperExclusive = MathHelper.NormalizeAngle(gradientAngle + angleRange / 2.0f),                    
                };                
                GradientAngle_DetectorValueRanges[gradientMagnitude, gradientAngleDegree] = detectorValueRange;
            }
        }        

        int width = (int)(Constants.RetinaImagePixelSize.Width / Constants.RetinaDetectorsDeltaPixels);
        int height = (int)(Constants.RetinaImagePixelSize.Height / Constants.RetinaDetectorsDeltaPixels);
        
        GradientMagnitude_Detectors = new DenseMatrix<SimpleDetector?>(width, height);
        //float[] detectorDensities_Accumulative = DistributionHelper.GetAccumulativeDistribution(templateDetectors.Select(d => d.Temp_Density).ToArray());
        //foreach (int dJ in Enumerable.Range(0, height))
        //    foreach (int dI in Enumerable.Range(0, width))
        //    {
        //        int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
        //        Detector templateDetector = templateDetectors[index];

        //        Detector detector = new()
        //        {
        //            Retina = this,
        //            DI = dI,
        //            DJ = dJ,
        //            CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
        //            CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
        //        };
        //        detector.Average = MathHelper.GetRandom(
        //            initializationRandom,
        //            templateDetector.Average,
        //            range: Constants.GradientMagnitudeDelta);
        //        detector.GradientAngle_Average = MathHelper.NormalizeAngle(MathHelper.GetRandom(
        //            initializationRandom,
        //            templateDetector.GradientAngle_Average,
        //            range: MathHelper.DegreesToRadians(Constants.GradientAngleDegreeDelta)));                
        //        detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
        //        //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

        //        GradientMagnitude_Detectors[dI, dJ] = detector;
        //    }

        GradientAngle_Detectors = new DenseMatrix<SimpleDetector?>(width, height);
        //float[] detectorDensities_Accumulative = DistributionHelper.GetAccumulativeDistribution(templateDetectors.Select(d => d.Temp_Density).ToArray());
        //foreach (int dJ in Enumerable.Range(0, height))
        //    foreach (int dI in Enumerable.Range(0, width))
        //    {
        //        int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
        //        Detector templateDetector = templateDetectors[index];

        //        Detector detector = new()
        //        {
        //            Retina = this,
        //            DI = dI,
        //            DJ = dJ,
        //            CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
        //            CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
        //        };
        //        detector.Average = MathHelper.GetRandom(
        //            initializationRandom,
        //            templateDetector.Average,
        //            range: Constants.GradientMagnitudeDelta);
        //        detector.GradientAngle_Average = MathHelper.NormalizeAngle(MathHelper.GetRandom(
        //            initializationRandom,
        //            templateDetector.GradientAngle_Average,
        //            range: MathHelper.DegreesToRadians(Constants.GradientAngleDegreeDelta)));
        //        detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
        //        //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

        //        GradientAngle_Detectors[dI, dJ] = detector;
        //    }

        FastList<GradientComplexDetector> templateDetectors = CalculateTemplate_GradientComplexDetectors(initializationRandom, Constants);
        GradientComplex_Detectors = new DenseMatrix<GradientComplexDetector?>(width, height);
        float[] detectorDensities_Accumulative = DistributionHelper.GetAccumulativeDistribution(templateDetectors.Select(d => d.Temp_Density).ToArray());
        foreach (int dJ in Enumerable.Range(0, height))
            foreach (int dI in Enumerable.Range(0, width))
            {
                int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
                GradientComplexDetector templateDetector = templateDetectors[index];

                GradientComplexDetector detector = new()
                {
                    Retina = this,
                    DI = dI,
                    DJ = dJ,
                    CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                    CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                };
                detector.GradientMagnitude_Average = MathHelper.GetRandom(
                    initializationRandom,
                    templateDetector.GradientMagnitude_Average,
                    range: Constants.GradientMagnitudeDelta);
                detector.GradientAngle_Average = MathHelper.NormalizeAngle(MathHelper.GetRandom(
                    initializationRandom,
                    templateDetector.GradientAngle_Average,
                    range: MathHelper.DegreesToRadians(Constants.GradientAngleDegreeDelta)));
                detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

                GradientComplex_Detectors[dI, dJ] = detector;
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
        foreach (int dJ in Enumerable.Range(0, GradientComplex_Detectors.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, GradientComplex_Detectors.Dimensions[0]))
            {
                Detector detector = GradientComplex_Detectors[dI, dJ]!;
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
            var gradientInPoint = MathHelper.GetInterpolatedGradient(
                retinaPoint.CenterXPixels,
                retinaPoint.CenterYPixels,
                eye_GradientMatrix);
            retinaPoint.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = (float)gradientInPoint.Magnitude;
            retinaPoint.FeaturesVector[FeaturesVector.GradientAngle_Index] = (float)gradientInPoint.Angle;
        }
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArrayOfUInt64(GradientMagnitude_AccumulativeDistribution);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientMagnitude_DetectorValueRanges, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientMagnitude_Detectors, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientAngle_DetectorValueRanges, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientAngle_Detectors, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientComplex_Detectors, writer, context);
            GradientMagnitude_Average_IdealPinwheel_MiniColumns.SerializeOwnedData(writer, context);
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
                    GradientMagnitude_DetectorValueRanges = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new DetectorValueRange());
                    GradientMagnitude_Detectors = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new SimpleDetector
                    {
                        Retina = this,
                        DI = dI,
                        DJ = dJ,
                        CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                        CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                    });
                    GradientAngle_DetectorValueRanges = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new DetectorValueRange());
                    GradientAngle_Detectors = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new SimpleDetector
                    {
                        Retina = this,
                        DI = dI,
                        DJ = dJ,
                        CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                        CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                    });
                    GradientComplex_Detectors = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new GradientComplexDetector
                    {
                        Retina = this,
                        DI = dI,
                        DJ = dJ,
                        CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                        CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                    });
                    GradientMagnitude_Average_IdealPinwheel_MiniColumns.DeserializeOwnedData(reader, context);
                    break;
            }
        }
    }

    #endregion

    private FastList<GradientComplexDetector> CalculateTemplate_GradientComplexDetectors(Random initializationRandom, IRetinaConstants constants)
    {           
        FastList<GradientComplexDetector> templateDetectors = new FastList<GradientComplexDetector>(10000);
        FastList<GradientComplexDetector> optimal_TemplateDetectors = new FastList<GradientComplexDetector>(10000);
        for (int gradientMagnitude = (int)GradientMagnitude_DetectorValueRanges[0, 0]!.LowerInclusive; gradientMagnitude < GradientMagnitude_DetectorValueRanges[GradientMagnitude_DetectorValueRanges.Dimensions[0] - 1, 0]!.UpperExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
            {                
                float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

                GradientComplexDetector detector = new GradientComplexDetector
                {
                    Retina = this,                    
                };
                detector.GradientMagnitude_Average = gradientMagnitude;
                detector.GradientAngle_Average = gradientAngle;
                detector.Temp_FeaturesVectorSamples = new FastList<FeaturesVectorSample>(300);
                detector.Temp_Density = 1.0f;
                templateDetectors.Add(detector);

                detector = new GradientComplexDetector
                {
                    Retina = this,
                };
                detector.GradientMagnitude_Average = gradientMagnitude;
                detector.GradientAngle_Average = gradientAngle;
                optimal_TemplateDetectors.Add(detector);
            }
        }        

        FastList<FeaturesVectorSample> featuresVectorSamples = new FastList<FeaturesVectorSample>((int)(constants.MaxGradientMagnitudeExclusive * 360 / (constants.GradientMagnitudeDelta * constants.GradientAngleDegreeDelta)));
        for (int gradientMagnitude = (int)constants.MinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
            {
                float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

                FeaturesVectorSample featuresVectorSample = new();
                featuresVectorSample.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = gradientMagnitude;
                featuresVectorSample.FeaturesVector[FeaturesVector.GradientAngle_Index] = gradientAngle;

                for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
                {
                    var templateGradientComplexDetector = templateDetectors[d_Index];                    
                    if (templateGradientComplexDetector.CalculateIsActivated(ref featuresVectorSample.FeaturesVector))
                    {
                        featuresVectorSample.Detectors.Add(templateGradientComplexDetector);
                        templateGradientComplexDetector.Temp_FeaturesVectorSamples.Add(featuresVectorSample);
                    }
                }

                if (featuresVectorSample.Detectors.Count > 0)
                    featuresVectorSamples.Add(featuresVectorSample);
                else
                    throw new InvalidOperationException();
            }
        }

        float min_ActivatedDeltaAbsMax = Single.MaxValue;
        for (; ; )
        {
            float activatedTotalAverage = 0.0f;
            for (int s_Index = 0; s_Index < featuresVectorSamples.Count; s_Index += 1)
            {
                var featuresVectorSample = featuresVectorSamples[s_Index];
                float activatedTotal = 0.0f;
                for (int d_Index = 0; d_Index < featuresVectorSample.Detectors.Count; d_Index += 1)
                {
                    var templateGradientComplexDetector = featuresVectorSample.Detectors[d_Index];
                    activatedTotal += templateGradientComplexDetector.Temp_Density;
                }
                featuresVectorSample.Temp_ActivatedTotal = activatedTotal;
                activatedTotalAverage += activatedTotal;
            }
            activatedTotalAverage /= featuresVectorSamples.Count;                        

            float activatedDeltaAbsMax = Single.MinValue;            
            for (int s_Index = 0; s_Index < featuresVectorSamples.Count; s_Index += 1)
            {
                var featuresVectorSample = featuresVectorSamples[s_Index];                
                float activatedDeltaAbs = MathF.Abs(featuresVectorSample.Temp_ActivatedTotal - activatedTotalAverage) / activatedTotalAverage;
                if (activatedDeltaAbs > activatedDeltaAbsMax)
                    activatedDeltaAbsMax = activatedDeltaAbs;
            }

            Logger.LogInformation($"Retina.CalculateGradientComplexDetectorDensities, activatedDelta_NormAbsMax: {activatedDeltaAbsMax}");

            if (activatedDeltaAbsMax > min_ActivatedDeltaAbsMax - 0.001f) // Working: 0.00001f
                break;

            if (activatedDeltaAbsMax < min_ActivatedDeltaAbsMax)
            {
                min_ActivatedDeltaAbsMax = activatedDeltaAbsMax;
                for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
                {
                    optimal_TemplateDetectors[d_Index].Temp_Density = templateDetectors[d_Index].Temp_Density;
                }
            }

            float densityAverage = 0.0f;
            for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
            {
                var templateGradientComplexDetector = templateDetectors[d_Index];
                if (templateGradientComplexDetector.Temp_FeaturesVectorSamples.Count == 0)
                {
                    templateGradientComplexDetector.Temp_Density = 0.0f;
                    continue;
                }
                float detector_K = 0.0f;
                for (int s_Index = 0; s_Index < templateGradientComplexDetector.Temp_FeaturesVectorSamples.Count; s_Index += 1)
                {
                    var featuresVectorSample = templateGradientComplexDetector.Temp_FeaturesVectorSamples[s_Index];
                    detector_K += featuresVectorSample.Temp_ActivatedTotal / activatedTotalAverage;
                }
                detector_K /= templateGradientComplexDetector.Temp_FeaturesVectorSamples.Count;

                if (detector_K == 0.0f)
                    throw new InvalidOperationException();

                templateGradientComplexDetector.Temp_Density /= detector_K;
                densityAverage += templateGradientComplexDetector.Temp_Density;
            }
            densityAverage /= templateDetectors.Count;

            for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
            {
                var templateGradientComplexDetector = templateDetectors[d_Index];
                templateGradientComplexDetector.Temp_Density /= densityAverage;
            }
        }

        return optimal_TemplateDetectors;
    }

    private void TestDetectorDensities(Random initializationRandom, FastList<GradientComplexDetector> templateDetectors, float[] detectorDensities_Accumulative)
    {
        GradientComplexDetector[] testDetectors = new GradientComplexDetector[Constants.MiniColumnVisibleDetectorsCount];
        for (int d_Index = 0; d_Index < testDetectors.Length; d_Index += 1)
        {
            int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
            GradientComplexDetector templateDetector = templateDetectors[index];

            GradientComplexDetector detector = new GradientComplexDetector()
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
                float angle = MathHelper.DegreesToRadians(gradientAngleDegree);
                FeaturesVector featuresVector = new();
                featuresVector[FeaturesVector.GradientMagnitude_Index] = gradientMagnitude;
                featuresVector[FeaturesVector.GradientAngle_Index] = angle;
                //{
                //    GradX = gradientMagnitude * Math.Cos(angle),
                //    GradY = gradientMagnitude * Math.Sin(angle),                    
                //};

                for (int d_Index = 0; d_Index < testDetectors.Length; d_Index += 1)
                {
                    var detector = testDetectors[d_Index];                    
                    if (detector.CalculateIsActivated(ref featuresVector))
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

/// <summary>
///     Диапазон какой-то детектируемой величины.
/// </summary>
public class DetectorValueRange : IOwnedDataSerializable
{    
    public float LowerInclusive;
    
    public float Average;

    public float UpperExclusive;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(LowerInclusive);
        writer.Write(Average);
        writer.Write(UpperExclusive);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        LowerInclusive = reader.ReadSingle();
        Average = reader.ReadSingle();
        UpperExclusive = reader.ReadSingle();
    }
}

public class FeaturesVectorSample
{
    public FeaturesVector FeaturesVector;

    public readonly FastList<Detector> Detectors = new(300);

    public float Temp_ActivatedTotal;
}

public class RetinaPoint
{
    public FeaturesVector FeaturesVector;

    public double CenterXPixels { get; init; }

    public double CenterYPixels { get; init; }
}

///// <summary>
/////     Диапазон какой-то детектируемой величины (возможно вектора).
///// </summary>
//public class DetectorValueRange : IOwnedDataSerializable
//{
//    public float[] LowerInclusive = null!;

//    public float[] Average = null!;

//    public float[] UpperExclusive = null!;

//    public void SerializeOwnedData(SerializationWriter writer, object? context)
//    {
//        writer.WriteArrayOfSingle(LowerInclusive);
//        writer.WriteArrayOfSingle(Average);
//        writer.WriteArrayOfSingle(UpperExclusive);
//    }

//    public void DeserializeOwnedData(SerializationReader reader, object? context)
//    {
//        LowerInclusive = reader.ReadArrayOfSingle();
//        Average = reader.ReadArrayOfSingle();
//        UpperExclusive = reader.ReadArrayOfSingle();
//    }
//}
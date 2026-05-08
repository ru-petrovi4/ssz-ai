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

        RetinaPoints = new DenseMatrix<RetinaPoint>((int)(Constants.RetinaImagePixelSize.Width / Constants.RetinaPointDeltaPixels), (int)(Constants.RetinaImagePixelSize.Height / Constants.RetinaPointDeltaPixels));
        RetinaPoints.CreateElementInstances((int x, int y) => new RetinaPoint()
        {
            CenterXPixels = x * Constants.RetinaPointDeltaPixels,
            CenterYPixels = y * Constants.RetinaPointDeltaPixels
        });
        ToCalculateRetinaPoints = new FastList<RetinaPoint>(RetinaPoints.Data);
    }

    #endregion

    #region public functions

    public readonly IRetinaConstants Constants;

    public readonly ILogger Logger;

    public readonly DenseMatrix<RetinaPoint> RetinaPoints;

    public FastList<RetinaPoint> ToCalculateRetinaPoints;

    public ulong[] GradientMagnitude_AccumulativeDistribution = null!;    

    /// <summary>
    ///     Диапазоны для детекторов в зависимости от модуля градиента (в градусах).
    /// </summary>
    public DenseMatrix<DetectorValueRange?> GradientMagnitude_DetectorValueRanges = new();

    /// <summary>
    ///     Диапазоны для детекторов в зависимости от угла градиента (в градусах).
    /// </summary>
    public DenseMatrix<DetectorValueRange?> GradientAngle_DetectorValueRanges = new();

    public DenseMatrix<DetectingPoint?> DetectingPoints_Matrix = new();

    public DenseMatrix<float> GradientMagnitude_Average_IdealPinwheel_MiniColumns = new();    

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

        GradientMagnitude_DetectorValueRanges = new DenseMatrix<DetectorValueRange?>((int)Constants.MaxGradientMagnitudeExclusive, 360);
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

        GradientAngle_DetectorValueRanges = new DenseMatrix<DetectorValueRange?>((int)Constants.MaxGradientMagnitudeExclusive, 360);
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

        GradientMagnitude_Average_IdealPinwheel_MiniColumns = new DenseMatrix<float>((int)Constants.MaxGradientMagnitudeExclusive, 360);
        for (int gradientMagnitude = 0; gradientMagnitude < GradientMagnitude_DetectorValueRanges.Dimensions[0]; gradientMagnitude += 1)
        {
            float samples = GradientMagnitude_AccumulativeDistribution[gradientMagnitude];
            float idealPinwheel_MiniColumns = samples / inIdealPinwheelMiniColumn_Samples;

            for (int gradientAngleDegree = 0; gradientAngleDegree < GradientMagnitude_DetectorValueRanges.Dimensions[1]; gradientAngleDegree += 1)
            {
                GradientMagnitude_Average_IdealPinwheel_MiniColumns[gradientMagnitude, gradientAngleDegree] = idealPinwheel_MiniColumns;
            }
        }

        int width = (int)(Constants.RetinaImagePixelSize.Width / Constants.DetectingPointDeltaPixels);
        int height = (int)(Constants.RetinaImagePixelSize.Height / Constants.DetectingPointDeltaPixels);
        
        DetectingPoints_Matrix = new DenseMatrix<DetectingPoint?>(width, height);

        const int simpleDetector_ActiveBitsCount = 5;        
        SampleDetectorsDistribution<SimpleDetector> gradientMagnitude_TemplateDetectors = 
            Calculate_GradientMagnitude_TemplateDetectors(initializationRandom, Constants);
        gradientMagnitude_TemplateDetectors.Density = CalculateDensity(initializationRandom, gradientMagnitude_TemplateDetectors, simpleDetector_ActiveBitsCount, width, height);

        SampleDetectorsDistribution<SimpleDetector> gradientAngle_TemplateDetectors = 
            Calculate_GradientAngle_TemplateDetectors(initializationRandom, Constants);
        gradientAngle_TemplateDetectors.Density = CalculateDensity(initializationRandom, gradientMagnitude_TemplateDetectors, simpleDetector_ActiveBitsCount, width, height);
        foreach (int dJ in Enumerable.Range(0, height))
            foreach (int dI in Enumerable.Range(0, width))
            {
                SimpleDetector gradientMagnitude_TemplateDetector = gradientMagnitude_TemplateDetectors.GetSampleDetector(initializationRandom);
                SimpleDetector gradientAngle_TemplateDetector = gradientAngle_TemplateDetectors.GetSampleDetector(initializationRandom);

                DetectingPoint detectingPoint = new()
                {
                    Retina = this,
                    DI = dI,
                    DJ = dJ,
                    CenterXPixels = dI * Constants.DetectingPointDeltaPixels,
                    CenterYPixels = dJ * Constants.DetectingPointDeltaPixels,

                    //GradientComplex_Detector = new(),
                };

                if (initializationRandom.NextSingle() < gradientMagnitude_TemplateDetectors.Density)
                    detectingPoint.GradientMagnitude_Detector = new(detectingPoint, FeaturesVector.GradientMagnitude_Index);

                if (initializationRandom.NextSingle() < gradientAngle_TemplateDetectors.Density)
                    detectingPoint.GradientAngle_Detector = new(detectingPoint, FeaturesVector.GradientAngle_Index);

                // Magnitude Average
                float gradientMagnitude_Detector_Average = MathHelper.GetRandom(
                        initializationRandom,
                        gradientMagnitude_TemplateDetector.Average,
                        range: Constants.GradientMagnitudeDelta);
                // Angle Average
                float gradientAngle_Detector_Average = MathHelper.NormalizeAngle(MathHelper.GetRandom(
                        initializationRandom,
                        gradientAngle_TemplateDetector.Average,
                        range: MathHelper.DegreesToRadians(Constants.GradientAngleDegreeDelta)));

                if (detectingPoint.GradientMagnitude_Detector is not null)
                {
                    detectingPoint.GradientMagnitude_Detector.Average = gradientMagnitude_Detector_Average;
                    detectingPoint.GradientMagnitude_Detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                }
                
                if (detectingPoint.GradientAngle_Detector is not null)
                {
                    detectingPoint.GradientAngle_Detector.Average = gradientAngle_Detector_Average;
                    detectingPoint.GradientAngle_Detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                }

                if (detectingPoint.GradientComplex_Detector is not null)
                {
                    detectingPoint.GradientComplex_Detector.GradientMagnitude_Average = gradientMagnitude_Detector_Average;
                    detectingPoint.GradientComplex_Detector.GradientAngle_Average = gradientAngle_Detector_Average;
                    detectingPoint.GradientComplex_Detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                }
                //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

                DetectingPoints_Matrix[dI, dJ] = detectingPoint;
            }

        //FastList<GradientComplexDetector> templateDetectors = CalculateTemplate_GradientComplexDetectors(initializationRandom, Constants);
        //float[] detectorDensities_Accumulative = DistributionHelper.GetAccumulativeDistribution(templateDetectors.Select(d => d.Temp_Density).ToArray());
        //foreach (int dJ in Enumerable.Range(0, height))
        //    foreach (int dI in Enumerable.Range(0, width))
        //    {
        //        int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
        //        GradientComplexDetector templateDetector = templateDetectors[index];

        //        DetectingPoint detectingPoint = new()
        //        {
        //            Retina = this,
        //            DI = dI,
        //            DJ = dJ,
        //            CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
        //            CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,

        //            GradientMagnitude_Detector = new(FeaturesVector.GradientMagnitude_Index),
        //            GradientAngle_Detector = new(FeaturesVector.GradientAngle_Index),
        //            GradientComplex_Detector = new(),
        //        };
        //        // Magnitude Average
        //        detectingPoint.GradientMagnitude_Detector.Average =
        //            detectingPoint.GradientComplex_Detector.GradientMagnitude_Average = MathHelper.GetRandom(
        //                initializationRandom,
        //                templateDetector.GradientMagnitude_Average,
        //                range: Constants.GradientMagnitudeDelta);
        //        // Angle Average
        //        detectingPoint.GradientAngle_Detector.Average =
        //            detectingPoint.GradientComplex_Detector.GradientAngle_Average = MathHelper.NormalizeAngle(MathHelper.GetRandom(
        //                initializationRandom,
        //                templateDetector.GradientAngle_Average,
        //                range: MathHelper.DegreesToRadians(Constants.GradientAngleDegreeDelta)));
        //        // BitIndexInHash
        //        detectingPoint.GradientMagnitude_Detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
        //        detectingPoint.GradientAngle_Detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
        //        detectingPoint.GradientComplex_Detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
        //        //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

        //        DetectingPoints_Matrix[dI, dJ] = detectingPoint;
        //    }

        //TestDetectorDensities(initializationRandom, templateDetectors, detectorDensities_Accumulative);        
    }    

    /// <summary>
    ///     Prepares for calculation after DeserializeOwnedData or GenerateOwnedData
    /// </summary>
    public void Prepare()
    {
        float detectorFieldOfViewRadiusPixels = Constants.DetectorFieldOfViewRadiusPixels;
        foreach (int dJ in Enumerable.Range(0, DetectingPoints_Matrix.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, DetectingPoints_Matrix.Dimensions[0]))
            {
                DetectingPoint detectingPoint = DetectingPoints_Matrix[dI, dJ]!;
                detectingPoint.Temp_RetinaPoints = new FastList<RetinaPoint>((int)(MathF.PI * (1 + detectorFieldOfViewRadiusPixels / Constants.RetinaPointDeltaPixels) * (1 + detectorFieldOfViewRadiusPixels / Constants.RetinaPointDeltaPixels)));

                for (int rpJ = (int)((detectingPoint.CenterYPixels - detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels); rpJ < (int)((detectingPoint.CenterYPixels + detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels) && rpJ < RetinaPoints.Dimensions[1]; rpJ += 1)
                    for (int rpI = (int)((detectingPoint.CenterXPixels - detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels); rpI < (int)((detectingPoint.CenterXPixels + detectorFieldOfViewRadiusPixels) / Constants.RetinaPointDeltaPixels) && rpI < RetinaPoints.Dimensions[0]; rpI += 1)
                    {
                        if (rpI < 0 || rpJ < 0)
                            continue;

                        RetinaPoint retinaPoint = RetinaPoints[rpI, rpJ]!;
                        double rPixels = Math.Sqrt((detectingPoint.CenterXPixels - retinaPoint.CenterXPixels) * (detectingPoint.CenterXPixels - retinaPoint.CenterXPixels) + (detectingPoint.CenterYPixels - retinaPoint.CenterYPixels) * (detectingPoint.CenterYPixels - retinaPoint.CenterYPixels));
                        if (rPixels < detectorFieldOfViewRadiusPixels)
                            detectingPoint.Temp_RetinaPoints.Add(retinaPoint);
                    }
            }       
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
        if (eye_GradientMatrix.HasTheSameValue)
        {
            var gradientInPoint = eye_GradientMatrix[0, 0];
            for (int rp_Index = 0; rp_Index < ToCalculateRetinaPoints.Count; rp_Index += 1)
            {
                RetinaPoint retinaPoint = ToCalculateRetinaPoints[rp_Index];                
                retinaPoint.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = (float)gradientInPoint.Magnitude;
                retinaPoint.FeaturesVector[FeaturesVector.GradientAngle_Index] = (float)gradientInPoint.Angle;
            }
        }
        else
        {
            for (int rp_Index = 0; rp_Index < ToCalculateRetinaPoints.Count; rp_Index += 1)
            {
                RetinaPoint retinaPoint = ToCalculateRetinaPoints[rp_Index];
                var gradientInPoint = MathHelper.GetInterpolatedGradient(
                    retinaPoint.CenterXPixels,
                    retinaPoint.CenterYPixels,
                    eye_GradientMatrix);
                retinaPoint.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = (float)gradientInPoint.Magnitude;
                retinaPoint.FeaturesVector[FeaturesVector.GradientAngle_Index] = (float)gradientInPoint.Angle;
            }
        }            
    }

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        using (writer.EnterBlock(1))
        {
            writer.WriteArrayOfUInt64(GradientMagnitude_AccumulativeDistribution);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientMagnitude_DetectorValueRanges, writer, context);            
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(GradientAngle_DetectorValueRanges, writer, context);            
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(DetectingPoints_Matrix, writer, context);
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
                    GradientAngle_DetectorValueRanges = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new DetectorValueRange());                    
                    DetectingPoints_Matrix = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new DetectingPoint
                    {
                        Retina = this,
                        DI = dI,
                        DJ = dJ,
                        CenterXPixels = dI * Constants.DetectingPointDeltaPixels,
                        CenterYPixels = dJ * Constants.DetectingPointDeltaPixels,
                    });
                    GradientMagnitude_Average_IdealPinwheel_MiniColumns.DeserializeOwnedData(reader, context);
                    break;
            }
        }
    }

    #endregion

    private float CalculateDensity(Random initializationRandom, SampleDetectorsDistribution<SimpleDetector> templateDetectors, int desiredActiveBitsCount, int width, int height)
    {
        DenseMatrix<GradientInPoint> eye_GradientMatrix = SobelOperator.ApplySobel(
            MathHelper.DegreesToRadians(Constants.TestGradientAngleDegrees),
            Constants.TestGradientMagnitude,
            Constants.TestGradientWidthRelative,
            Constants.TestGradientPositionRelative,
            Constants.RetinaImagePixelSize.Width,
            Constants.RetinaImagePixelSize.Height);
        CalculateRetinaPoints(eye_GradientMatrix);

        int activatedCount = 0;
        
        foreach (int _ in Enumerable.Range(0, Constants.MiniColumnVisibleDetectingPointsCount))
        {
            SimpleDetector templateDetector = templateDetectors.GetSampleDetector(initializationRandom);

            templateDetector.Temp_IsActivated = templateDetector.CalculateIsActivated();
            if (templateDetector.Temp_IsActivated)
                activatedCount += 1;
        }        

        return (float)desiredActiveBitsCount / activatedCount;
    }

    private SampleDetectorsDistribution<SimpleDetector> Calculate_GradientMagnitude_TemplateDetectors(Random initializationRandom, IRetinaConstants constants)
    {
        SampleDetectorsDistribution<SimpleDetector> sampleDetectorsDistribution = new()
        {
            DetectorInfos = new FastList<SampleDetectorsDistribution<SimpleDetector>.DetectorInfo>(10000),
        };

        var templateDetector_RetinaPoint = ToCalculateRetinaPoints[0];

        DetectingPoint detectingPoint = new DetectingPoint()
        {
            Retina = this,
            Temp_RetinaPoints = new FastList<RetinaPoint>() { templateDetector_RetinaPoint }
        };
        
        for (int gradientMagnitude = (int)GradientMagnitude_DetectorValueRanges[0, 0]!.LowerInclusive; 
                gradientMagnitude < GradientMagnitude_DetectorValueRanges[GradientMagnitude_DetectorValueRanges.Dimensions[0] - 1, 0]!.UpperExclusive; 
                gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {            
            SampleDetectorsDistribution<SimpleDetector>.DetectorInfo tdi = new()
            {
                Detector = new SimpleDetector(detectingPoint, FeaturesVector.GradientMagnitude_Index),
                Density = 1.0f,
                FeaturesVectorSamples = new FastList<FeaturesVectorSample<SimpleDetector>>(16)
            };
            tdi.Detector.Average = gradientMagnitude;
            sampleDetectorsDistribution.DetectorInfos.Add(tdi);
        }

        FastList<FeaturesVectorSample<SimpleDetector>> featuresVectorSamples = new FastList<FeaturesVectorSample<SimpleDetector>>((int)(constants.MaxGradientMagnitudeExclusive / constants.GradientMagnitudeDelta));
        for (int gradientMagnitude = (int)constants.MinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
        {
            FeaturesVectorSample<SimpleDetector> featuresVectorSample = new();
            featuresVectorSample.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = gradientMagnitude;
            featuresVectorSample.FeaturesVector[FeaturesVector.GradientAngle_Index] = 0;

            templateDetector_RetinaPoint.FeaturesVector = featuresVectorSample.FeaturesVector;

            for (int d_Index = 0; d_Index < sampleDetectorsDistribution.DetectorInfos.Count; d_Index += 1)
            {
                var tdi = sampleDetectorsDistribution.DetectorInfos[d_Index];                
                if (tdi.Detector.CalculateIsActivated())
                {
                    featuresVectorSample.DetectorInfos.Add(tdi);
                    tdi.FeaturesVectorSamples.Add(featuresVectorSample);
                }
            }

            if (featuresVectorSample.DetectorInfos.Count > 0)
                featuresVectorSamples.Add(featuresVectorSample);
            else
                throw new InvalidOperationException();
        }

        return Calculate_TemplateDetectors(initializationRandom, constants, sampleDetectorsDistribution, featuresVectorSamples);        
    }

    private SampleDetectorsDistribution<SimpleDetector> Calculate_GradientAngle_TemplateDetectors(Random initializationRandom, IRetinaConstants constants)
    {
        SampleDetectorsDistribution<SimpleDetector> sampleDetectorsDistribution = new()
        {
            DetectorInfos = new FastList<SampleDetectorsDistribution<SimpleDetector>.DetectorInfo>(10000),
        };

        var templateDetector_RetinaPoint = ToCalculateRetinaPoints[0];

        DetectingPoint detectingPoint = new DetectingPoint()
        {
            Retina = this,
            Temp_RetinaPoints = new FastList<RetinaPoint>() { templateDetector_RetinaPoint }
        };

        for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
        {
            float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

            SampleDetectorsDistribution<SimpleDetector>.DetectorInfo tdi = new()
            {
                Detector = new SimpleDetector(detectingPoint, FeaturesVector.GradientAngle_Index),
                Density = 1.0f,
                FeaturesVectorSamples = new FastList<FeaturesVectorSample<SimpleDetector>>(16)
            };
            tdi.Detector.Average = gradientAngle;
            sampleDetectorsDistribution.DetectorInfos.Add(tdi);
        }

        FastList<FeaturesVectorSample<SimpleDetector>> featuresVectorSamples = new FastList<FeaturesVectorSample<SimpleDetector>>((int)(360 / constants.GradientAngleDegreeDelta));
        for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
        {
            float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

            FeaturesVectorSample<SimpleDetector> featuresVectorSample = new();
            featuresVectorSample.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = constants.MaxGradientMagnitudeExclusive / 2.0f;
            featuresVectorSample.FeaturesVector[FeaturesVector.GradientAngle_Index] = gradientAngle;

            templateDetector_RetinaPoint.FeaturesVector = featuresVectorSample.FeaturesVector;

            for (int d_Index = 0; d_Index < sampleDetectorsDistribution.DetectorInfos.Count; d_Index += 1)
            {
                var tdi = sampleDetectorsDistribution.DetectorInfos[d_Index];
                if (tdi.Detector.CalculateIsActivated())
                {
                    featuresVectorSample.DetectorInfos.Add(tdi);
                    tdi.FeaturesVectorSamples.Add(featuresVectorSample);
                }
            }

            if (featuresVectorSample.DetectorInfos.Count > 0)
                featuresVectorSamples.Add(featuresVectorSample);
            else
                throw new InvalidOperationException();
        }

        return Calculate_TemplateDetectors(initializationRandom, constants, sampleDetectorsDistribution, featuresVectorSamples);
    }

    private SampleDetectorsDistribution<SimpleDetector> Calculate_TemplateDetectors(Random initializationRandom, IRetinaConstants constants, SampleDetectorsDistribution<SimpleDetector> sampleDetectorsDistribution, FastList<FeaturesVectorSample<SimpleDetector>> featuresVectorSamples)
    {
        float min_ActivatedDeltaAbsMax = Single.MaxValue;
        for (; ; )
        {
            float activatedTotalAverage = 0.0f;
            for (int s_Index = 0; s_Index < featuresVectorSamples.Count; s_Index += 1)
            {
                var featuresVectorSample = featuresVectorSamples[s_Index];
                float activatedTotal = 0.0f;
                for (int d_Index = 0; d_Index < featuresVectorSample.DetectorInfos.Count; d_Index += 1)
                {
                    var detectorInfo = featuresVectorSample.DetectorInfos[d_Index];
                    activatedTotal += detectorInfo.Density;
                }
                featuresVectorSample.ActivatedTotal = activatedTotal;
                activatedTotalAverage += activatedTotal;
            }
            activatedTotalAverage /= featuresVectorSamples.Count;            

            float activatedDeltaAbsMax = Single.MinValue;
            for (int s_Index = 0; s_Index < featuresVectorSamples.Count; s_Index += 1)
            {
                var featuresVectorSample = featuresVectorSamples[s_Index];
                float activatedDeltaAbs = MathF.Abs(featuresVectorSample.ActivatedTotal - activatedTotalAverage) / activatedTotalAverage;
                if (activatedDeltaAbs > activatedDeltaAbsMax)
                    activatedDeltaAbsMax = activatedDeltaAbs;
            }

            Logger.LogInformation($"Retina.Calculate_TemplateDetectors, activatedDelta_NormAbsMax: {activatedDeltaAbsMax}");

            if (activatedDeltaAbsMax > min_ActivatedDeltaAbsMax - 0.001f) // Working: 0.00001f
                break;

            if (activatedDeltaAbsMax < min_ActivatedDeltaAbsMax)
                min_ActivatedDeltaAbsMax = activatedDeltaAbsMax;

            float densityAverage = 0.0f;
            for (int d_Index = 0; d_Index < sampleDetectorsDistribution.DetectorInfos.Count; d_Index += 1)
            {
                var detectorInfo = sampleDetectorsDistribution.DetectorInfos[d_Index];
                if (detectorInfo.FeaturesVectorSamples.Count == 0)
                {
                    detectorInfo.Density = 0.0f;
                    continue;
                }
                float detector_K = 0.0f;
                for (int s_Index = 0; s_Index < detectorInfo.FeaturesVectorSamples.Count; s_Index += 1)
                {
                    var featuresVectorSample = detectorInfo.FeaturesVectorSamples[s_Index];
                    detector_K += featuresVectorSample.ActivatedTotal / activatedTotalAverage;
                }
                detector_K /= detectorInfo.FeaturesVectorSamples.Count;

                if (detector_K == 0.0f)
                    throw new InvalidOperationException();

                detectorInfo.Density /= detector_K;
                densityAverage += detectorInfo.Density;
            }
            densityAverage /= sampleDetectorsDistribution.DetectorInfos.Count;

            for (int d_Index = 0; d_Index < sampleDetectorsDistribution.DetectorInfos.Count; d_Index += 1)
            {
                var detectorInfo = sampleDetectorsDistribution.DetectorInfos[d_Index];
                detectorInfo.Density /= densityAverage;
            }
        }

        sampleDetectorsDistribution.DetectorDensities_Accumulative = DistributionHelper.GetAccumulativeDistribution(
            sampleDetectorsDistribution.DetectorInfos.Select(tdi => tdi.Density).ToArray());        

        return sampleDetectorsDistribution;
    }

    //private FastList<GradientComplexDetector> CalculateTemplate_GradientComplexDetectors(Random initializationRandom, IRetinaConstants constants)
    //{           
    //    FastList<GradientComplexDetector> templateDetectors = new FastList<GradientComplexDetector>(10000);
    //    FastList<GradientComplexDetector> optimal_TemplateDetectors = new FastList<GradientComplexDetector>(10000);
    //    for (int gradientMagnitude = (int)GradientMagnitude_DetectorValueRanges[0, 0]!.LowerInclusive; gradientMagnitude < GradientMagnitude_DetectorValueRanges[GradientMagnitude_DetectorValueRanges.Dimensions[0] - 1, 0]!.UpperExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
    //    {
    //        for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
    //        {                
    //            float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

    //            GradientComplexDetector detector = new GradientComplexDetector();
    //            detector.GradientMagnitude_Average = gradientMagnitude;
    //            detector.GradientAngle_Average = gradientAngle;
    //            detector.Temp_FeaturesVectorSamples = new FastList<FeaturesVectorSample>(300);
    //            detector.Temp_Density = 1.0f;
    //            templateDetectors.Add(detector);

    //            detector = new GradientComplexDetector();
    //            detector.GradientMagnitude_Average = gradientMagnitude;
    //            detector.GradientAngle_Average = gradientAngle;
    //            optimal_TemplateDetectors.Add(detector);
    //        }
    //    }        

    //    FastList<FeaturesVectorSample> featuresVectorSamples = new FastList<FeaturesVectorSample>((int)(constants.MaxGradientMagnitudeExclusive * 360 / (constants.GradientMagnitudeDelta * constants.GradientAngleDegreeDelta)));
    //    for (int gradientMagnitude = (int)constants.MinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
    //    {
    //        for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)constants.GradientAngleDegreeDelta)
    //        {
    //            float gradientAngle = MathHelper.DegreesToRadians(gradientAngleDegree);

    //            FeaturesVectorSample featuresVectorSample = new();
    //            featuresVectorSample.FeaturesVector[FeaturesVector.GradientMagnitude_Index] = gradientMagnitude;
    //            featuresVectorSample.FeaturesVector[FeaturesVector.GradientAngle_Index] = gradientAngle;

    //            for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
    //            {
    //                var templateGradientComplexDetector = templateDetectors[d_Index];                    
    //                if (templateGradientComplexDetector.CalculateIsActivated(ref featuresVectorSample.FeaturesVector))
    //                {
    //                    featuresVectorSample.Detectors.Add(templateGradientComplexDetector);
    //                    templateGradientComplexDetector.Temp_FeaturesVectorSamples.Add(featuresVectorSample);
    //                }
    //            }

    //            if (featuresVectorSample.Detectors.Count > 0)
    //                featuresVectorSamples.Add(featuresVectorSample);
    //            else
    //                throw new InvalidOperationException();
    //        }
    //    }

    //    float min_ActivatedDeltaAbsMax = Single.MaxValue;
    //    for (; ; )
    //    {
    //        float activatedTotalAverage = 0.0f;
    //        for (int s_Index = 0; s_Index < featuresVectorSamples.Count; s_Index += 1)
    //        {
    //            var featuresVectorSample = featuresVectorSamples[s_Index];
    //            float activatedTotal = 0.0f;
    //            for (int d_Index = 0; d_Index < featuresVectorSample.Detectors.Count; d_Index += 1)
    //            {
    //                var templateGradientComplexDetector = featuresVectorSample.Detectors[d_Index];
    //                activatedTotal += templateGradientComplexDetector.Temp_Density;
    //            }
    //            featuresVectorSample.Temp_ActivatedTotal = activatedTotal;
    //            activatedTotalAverage += activatedTotal;
    //        }
    //        activatedTotalAverage /= featuresVectorSamples.Count;                        

    //        float activatedDeltaAbsMax = Single.MinValue;            
    //        for (int s_Index = 0; s_Index < featuresVectorSamples.Count; s_Index += 1)
    //        {
    //            var featuresVectorSample = featuresVectorSamples[s_Index];                
    //            float activatedDeltaAbs = MathF.Abs(featuresVectorSample.Temp_ActivatedTotal - activatedTotalAverage) / activatedTotalAverage;
    //            if (activatedDeltaAbs > activatedDeltaAbsMax)
    //                activatedDeltaAbsMax = activatedDeltaAbs;
    //        }

    //        Logger.LogInformation($"Retina.CalculateGradientComplexDetectorDensities, activatedDelta_NormAbsMax: {activatedDeltaAbsMax}");

    //        if (activatedDeltaAbsMax > min_ActivatedDeltaAbsMax - 0.001f) // Working: 0.00001f
    //            break;

    //        if (activatedDeltaAbsMax < min_ActivatedDeltaAbsMax)
    //        {
    //            min_ActivatedDeltaAbsMax = activatedDeltaAbsMax;
    //            for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
    //            {
    //                optimal_TemplateDetectors[d_Index].Temp_Density = templateDetectors[d_Index].Temp_Density;
    //            }
    //        }

    //        float densityAverage = 0.0f;
    //        for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
    //        {
    //            var templateGradientComplexDetector = templateDetectors[d_Index];
    //            if (templateGradientComplexDetector.Temp_FeaturesVectorSamples.Count == 0)
    //            {
    //                templateGradientComplexDetector.Temp_Density = 0.0f;
    //                continue;
    //            }
    //            float detector_K = 0.0f;
    //            for (int s_Index = 0; s_Index < templateGradientComplexDetector.Temp_FeaturesVectorSamples.Count; s_Index += 1)
    //            {
    //                var featuresVectorSample = templateGradientComplexDetector.Temp_FeaturesVectorSamples[s_Index];
    //                detector_K += featuresVectorSample.Temp_ActivatedTotal / activatedTotalAverage;
    //            }
    //            detector_K /= templateGradientComplexDetector.Temp_FeaturesVectorSamples.Count;

    //            if (detector_K == 0.0f)
    //                throw new InvalidOperationException();

    //            templateGradientComplexDetector.Temp_Density /= detector_K;
    //            densityAverage += templateGradientComplexDetector.Temp_Density;
    //        }
    //        densityAverage /= templateDetectors.Count;

    //        for (int d_Index = 0; d_Index < templateDetectors.Count; d_Index += 1)
    //        {
    //            var templateGradientComplexDetector = templateDetectors[d_Index];
    //            templateGradientComplexDetector.Temp_Density /= densityAverage;
    //        }
    //    }

    //    return optimal_TemplateDetectors;
    //}

    private void TestDetectorDensities(Random initializationRandom, FastList<GradientComplexDetector> templateDetectors, float[] detectorDensities_Accumulative)
    {
        //GradientComplexDetector[] testDetectors = new GradientComplexDetector[Constants.MiniColumnVisibleDetectorsCount];
        //for (int d_Index = 0; d_Index < testDetectors.Length; d_Index += 1)
        //{
        //    int index = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative);
        //    GradientComplexDetector templateDetector = templateDetectors[index];

        //    GradientComplexDetector detector = new GradientComplexDetector()
        //    {
        //        Retina = this,
        //    };
        //    detector.GradientMagnitude_Average = templateDetector.GradientMagnitude_Average;
        //    detector.GradientAngle_Average = templateDetector.GradientAngle_Average;
        //    detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);

        //    testDetectors[d_Index] = detector;
        //}        

        //DataToDisplayHolder dataToDisplayHolder = DataToDisplayHolder.Instance;
        //dataToDisplayHolder.DistributionXMin = 0.0f;
        //dataToDisplayHolder.DistributionXMax = Constants.MaxGradientMagnitudeExclusive / Constants.GradientMagnitudeDelta;
        //dataToDisplayHolder.Distribution = new ulong[(int)(Constants.MaxGradientMagnitudeExclusive / Constants.GradientMagnitudeDelta) + 1];

        //for (int gradientMagnitude = (int)Constants.MinGradientMagnitudeInclusive; gradientMagnitude < Constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)Constants.GradientMagnitudeDelta)
        //{
        //    int activatedCount = 0;
        //    for (int gradientAngleDegree = 0; gradientAngleDegree < 360; gradientAngleDegree += (int)Constants.GradientAngleDegreeDelta)
        //    {
        //        float angle = MathHelper.DegreesToRadians(gradientAngleDegree);
        //        FeaturesVector featuresVector = new();
        //        featuresVector[FeaturesVector.GradientMagnitude_Index] = gradientMagnitude;
        //        featuresVector[FeaturesVector.GradientAngle_Index] = angle;
        //        //{
        //        //    GradX = gradientMagnitude * Math.Cos(angle),
        //        //    GradY = gradientMagnitude * Math.Sin(angle),                    
        //        //};

        //        for (int d_Index = 0; d_Index < testDetectors.Length; d_Index += 1)
        //        {
        //            var detector = testDetectors[d_Index];                    
        //            if (detector.CalculateIsActivated(ref featuresVector))
        //            {
        //                activatedCount += 1;
        //                detector.Temp_IsActivatedCount += 1;
        //            }
        //        }
        //    }
        //    dataToDisplayHolder.Distribution[(int)(gradientMagnitude / Constants.GradientMagnitudeDelta)] = (ulong)(activatedCount * Constants.GradientAngleDegreeDelta / 360);
        //}

        //Logger.LogInformation($"TestDetectorDensities: " + String.Join(", ", dataToDisplayHolder.Distribution.Select((v, i) => $"{v}")));
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

public class FeaturesVectorSample<TDetector>
    where TDetector : Detector
{
    public FeaturesVector FeaturesVector;

    public readonly FastList<SampleDetectorsDistribution<TDetector>.DetectorInfo> DetectorInfos = new(300);

    public float ActivatedTotal;
}

public class RetinaPoint
{
    public FeaturesVector FeaturesVector;

    public double CenterXPixels { get; init; }

    public double CenterYPixels { get; init; }
}

public class SampleDetectorsDistribution<TDetector>
    where TDetector : Detector
{
    public FastList<DetectorInfo> DetectorInfos = null!;

    public float[] DetectorDensities_Accumulative = null!;

    /// <summary>
    ///    [0.0..1.0]
    /// </summary>
    public float Density;    

    public TDetector GetSampleDetector(Random random)
    {
        int index = DistributionHelper.GetRandom(random, DetectorDensities_Accumulative);
        return DetectorInfos[index].Detector;
    }

    public class DetectorInfo
    {
        public TDetector Detector = null!;

        public int IsActivatedCount;

        public float Density;

        public FastList<FeaturesVectorSample<TDetector>> FeaturesVectorSamples = null!;
    }
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
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

	public FastList<GradientMagnitudeRange?> IdealPinwheel_GradientMagnitudeRanges = new();

	public DenseMatrix<Detector?> Detectors = new();	

	/// <summary>
	///     Диапазоны для детекторов в зависимости от модуля градиента и угла градиента (в градусах).
	/// </summary>
	public DenseMatrix<DetectorRange?> DetectorRanges = new();	

	//public float[] AngleAccumulativeDistribution = null!;

	/// <summary>
	///     Generates model data after construction.
	/// </summary>
	public void GenerateOwnedData(Random initializationRandom, GradientDistribution gradientDistribution)
    {
        int idealPinwheel_GradientMagnitudeRanges_Count = Constants.HyperColumnDefinedRadius_MiniColumns + 1;
        IdealPinwheel_GradientMagnitudeRanges = new FastList<GradientMagnitudeRange?>(idealPinwheel_GradientMagnitudeRanges_Count);
        ulong[] gradientMagnitude_AccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
        ulong totalSamples = gradientMagnitude_AccumulativeDistribution[^1];
        ulong samplesInIdealPinwheelMiniColumn = totalSamples / (ulong)idealPinwheel_GradientMagnitudeRanges_Count;        
        int prev_GradientMagnitudeRange_UpperLimit = 0;        
        for (int range_Index = 0; range_Index < idealPinwheel_GradientMagnitudeRanges_Count; range_Index += 1)
        {
            ulong upperLimit_SamplesCount = samplesInIdealPinwheelMiniColumn * (ulong)(range_Index + 1);

            int gradientMagnitudeRange_UpperLimit = DistributionHelper.GetIndex(upperLimit_SamplesCount, gradientMagnitude_AccumulativeDistribution);

            // Переделать на только Average
            IdealPinwheel_GradientMagnitudeRanges.Add(new GradientMagnitudeRange
            {
                Average_GradientMagnitude = (gradientMagnitudeRange_UpperLimit + prev_GradientMagnitudeRange_UpperLimit) / 2,
                HalfRange_GradientMagnitude = (gradientMagnitudeRange_UpperLimit - prev_GradientMagnitudeRange_UpperLimit) / 2,
            });

            prev_GradientMagnitudeRange_UpperLimit = gradientMagnitudeRange_UpperLimit;
        }
        
        float gradientMagnitudeRange_Samples = 5.0f * samplesInIdealPinwheelMiniColumn;        

        DetectorRanges = new DenseMatrix<DetectorRange?>(Constants.MaxGradientMagnitudeExclusive, 360);                
        
        for (int gradientMagnitude = 0; gradientMagnitude < DetectorRanges.Dimensions[0]; gradientMagnitude += 1)
        {
            float idealPinwheel_MiniColumns = gradientMagnitude_AccumulativeDistribution[gradientMagnitude] / samplesInIdealPinwheelMiniColumn;
            
            float fullCircle_MiniColuns = 2.0f * MathF.PI * idealPinwheel_MiniColumns;
            float gradientAngleRange_MiniColumns = 5.0f;
            float angleRange = 2.0f * MathF.PI * gradientAngleRange_MiniColumns / fullCircle_MiniColuns;
            
            if (Single.IsInfinity(angleRange) || angleRange > 2 * MathF.PI)
                angleRange = 2 * MathF.PI;

            int gradientMagnitudeRange_LowerLimit = DistributionHelper.GetIndex(gradientMagnitude_AccumulativeDistribution[gradientMagnitude] - (ulong)(gradientMagnitudeRange_Samples / 2), gradientMagnitude_AccumulativeDistribution);
            int gradientMagnitudeRange_UpperLimit = DistributionHelper.GetIndex(gradientMagnitude_AccumulativeDistribution[gradientMagnitude] + (ulong)(gradientMagnitudeRange_Samples / 2), gradientMagnitude_AccumulativeDistribution);            

            for (int gradientAngleDegree = 0; gradientAngleDegree < DetectorRanges.Dimensions[1]; gradientAngleDegree += 1)            
            {
                DetectorRanges[gradientMagnitude, gradientAngleDegree] = new DetectorRange
                {
                    HalfRange_GradientMagnitude = Math.Max(gradientMagnitudeRange_UpperLimit - gradientMagnitude, gradientMagnitude - gradientMagnitudeRange_LowerLimit),
                    HalfRange_GradientAngle = angleRange / 2
                };
            }
        }

        // Плотность детекторов, в зависимости от модуля градиента и угла градиента (в градусах) детектора.
        MatrixFloat_ColumnMajor detectorDensities = new MatrixFloat_ColumnMajor((int)((Constants.MaxGradientMagnitudeExclusive +
            DetectorRanges[DetectorRanges.Dimensions[0] - 1, 0]!.HalfRange_GradientMagnitude) / Constants.GradientMagnitudeDelta), (int)(360 / Constants.GradientAngleDegreeDelta));

        CalculateDetectorDensities(initializationRandom, detectorDensities, Constants);
        
        Detectors = new DenseMatrix<Detector?>(
            (int)Math.Round(Constants.RetinaImagePixelSize.Width / Constants.RetinaDetectorsDeltaPixels, 0),
            (int)Math.Round(Constants.RetinaImagePixelSize.Height / Constants.RetinaDetectorsDeltaPixels, 0));        

        MatrixFloat_ColumnMajor detectorDensities_Accumulative = new MatrixFloat_ColumnMajor(
            DistributionHelper.GetAccumulativeDistribution(detectorDensities.Data),
            [detectorDensities.Dimensions[0], detectorDensities.Dimensions[1]]);

        foreach (int dJ in Enumerable.Range(0, Detectors.Dimensions[1]))
            foreach (int dI in Enumerable.Range(0, Detectors.Dimensions[0]))
            {
                int dataIndex = DistributionHelper.GetRandom(initializationRandom, detectorDensities_Accumulative.Data);
                var indices = detectorDensities_Accumulative.GetIndices(dataIndex);

                Detector detector = new()
                {
                    DI = dI,
                    DJ = dJ,
                    CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                    CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                };
                
                detector.Average_GradientMagnitude = indices.I * Constants.GradientMagnitudeDelta;
                detector.Average_GradientAngle = MathHelper.DegreesToRadians(indices.J * Constants.GradientAngleDegreeDelta);                
                detector.BitIndexInHash = initializationRandom.Next(Constants.HashLength);
                //dataToDisplayHolder.Distribution[(int)MathHelper.RadiansToDegrees(detector.GradientAngleMax)] += 1;

                Detectors[dI, dJ] = detector;
            }

        TestDetectorDensities(initializationRandom, detectorDensities_Accumulative, Constants);        
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
			Helpers.SerializationHelper.SerializeOwnedData_FastList(IdealPinwheel_GradientMagnitudeRanges, writer, context);
			Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(Detectors, writer, context);
            Helpers.SerializationHelper.SerializeOwnedData_DenseMatrix(DetectorRanges, writer, context);
        }
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        using (Block block = reader.EnterBlock())
        {
            switch (block.Version)
            { 
                case 1:
					IdealPinwheel_GradientMagnitudeRanges = Helpers.SerializationHelper.DeserializeOwnedData_FastList(reader, context, (int index) => new GradientMagnitudeRange());
					Detectors = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new Detector
                    {
                        DI = dI,
                        DJ = dJ,
                        CenterXPixels = dI * Constants.RetinaDetectorsDeltaPixels,
                        CenterYPixels = dJ * Constants.RetinaDetectorsDeltaPixels,
                    });
                    DetectorRanges = Helpers.SerializationHelper.DeserializeOwnedData_DenseMatrix(reader, context, (int dI, int dJ) => new DetectorRange());                    
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
                detector.Average_GradientMagnitude = dI * constants.GradientMagnitudeDelta;
                detector.Average_GradientAngle = MathHelper.DegreesToRadians(dJ * constants.GradientAngleDegreeDelta);
                detector.Temp_Samples = new FastList<Sample>(300);

                //detector.AverageGradientMagnitude = initializationRandom.NextSingle() * detectorDensities.Dimensions[0];
                //detector.AverageGradientAngle = MathHelper.DegreesToRadians(initializationRandom.Next(detectorDensities.Dimensions[1]));            

                testDetectors[dI, dJ] = detector;
            }

        FastList<Sample> samples = new FastList<Sample>((int)(((constants.MaxGradientMagnitudeExclusive + 1) * 361) / (constants.GradientMagnitudeDelta * constants.GradientAngleDegreeDelta)));
        for (int gradientMagnitude = (int)constants.MinGradientMagnitudeInclusive; gradientMagnitude < constants.MaxGradientMagnitudeExclusive; gradientMagnitude += (int)constants.GradientMagnitudeDelta)
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

            if (activatedDelta_NormAbsMax < 0.55)
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

            detector.Average_GradientMagnitude = indices.I * constants.GradientMagnitudeDelta;
            detector.Average_GradientAngle = MathHelper.DegreesToRadians(indices.J * constants.GradientAngleDegreeDelta);
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

public class Detector : IOwnedDataSerializable
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
    
    public float Average_GradientMagnitude;        

    /// <summary>
    ///     [-pi, pi]
    /// </summary>
    public float Average_GradientAngle;        

    public int BitIndexInHash;

    public bool Temp_IsActivated;

    public int Temp_IsActivatedCount;

    public GradientInPoint Temp_GradientInPoint;

    public FastList<Sample> Temp_Samples = null!;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(Average_GradientMagnitude);
        writer.Write(Average_GradientAngle);
        writer.Write(BitIndexInHash);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        Average_GradientMagnitude = reader.ReadSingle();
        Average_GradientAngle = reader.ReadSingle();
        BitIndexInHash = reader.ReadInt32();
    }

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

        if (gradientInPoint.Magnitude < constants.MinGradientMagnitudeInclusive || 
                gradientInPoint.Magnitude >= constants.MaxGradientMagnitudeExclusive)
        {
            Temp_IsActivated = false;
            return;
        }

        DetectorRange detectorRange = retina.DetectorRanges[(int)gradientInPoint.Magnitude, (int)MathHelper.RadiansToDegrees((float)gradientInPoint.Angle)]!;

        bool activated = (gradientInPoint.Magnitude >= Average_GradientMagnitude - detectorRange.HalfRange_GradientMagnitude) &&
            (gradientInPoint.Magnitude < Average_GradientMagnitude + detectorRange.HalfRange_GradientMagnitude);
        if (!activated)
        {
            Temp_IsActivated = false;
            return;
        }

        // [-pi, pi)
        float gradientAngleMin = MathHelper.NormalizeAngle(Average_GradientAngle - detectorRange.HalfRange_GradientAngle);
        float gradientAngleMax = MathHelper.NormalizeAngle(Average_GradientAngle + detectorRange.HalfRange_GradientAngle);
        if (gradientAngleMax > gradientAngleMin + 0.01f)
            activated = (gradientInPoint.Angle >= gradientAngleMin) && (gradientInPoint.Angle < gradientAngleMax);
        else
            activated = (gradientInPoint.Angle >= gradientAngleMin) || (gradientInPoint.Angle < gradientAngleMax);
        Temp_IsActivated = activated;
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
///     Диапазон для детектора.
/// </summary>
public class GradientMagnitudeRange : IOwnedDataSerializable
{
	/// <summary>
	///     Половина диапазона угла градиента.
	/// </summary>
	public float Average_GradientMagnitude;

	/// <summary>
	///     Половина диапазона модуля градиента.
	/// </summary>
	public float HalfRange_GradientMagnitude;	

	public void SerializeOwnedData(SerializationWriter writer, object? context)
	{
		writer.Write(Average_GradientMagnitude);
		writer.Write(HalfRange_GradientMagnitude);
	}

	public void DeserializeOwnedData(SerializationReader reader, object? context)
	{
		Average_GradientMagnitude = reader.ReadSingle();
		HalfRange_GradientMagnitude = reader.ReadSingle();
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
    public float HalfRange_GradientMagnitude;

    /// <summary>
    ///     Половина диапазона угла градиента.
    /// </summary>
    public float HalfRange_GradientAngle;

    public void SerializeOwnedData(SerializationWriter writer, object? context)
    {
        writer.Write(HalfRange_GradientMagnitude);
        writer.Write(HalfRange_GradientAngle);
    }

    public void DeserializeOwnedData(SerializationReader reader, object? context)
    {
        HalfRange_GradientMagnitude = reader.ReadSingle();
        HalfRange_GradientAngle = reader.ReadSingle();
    }
}

public class Sample
{    
    public int gradientMagnitude;
    
    public int gradientAngleDegree;

    public readonly FastList<Detector> Detectors = new(300);

    public float Temp_ActivatedTotal;
}

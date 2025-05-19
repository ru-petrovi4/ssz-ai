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
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Models
{
    public class Retina : ISerializableModelObject
    {
        #region construction and destruction

        public Retina(IConstants constants, int imageWidth, int imageHeight)
        {
            Detectors = new DenseMatrix<Detector>((int)(imageWidth / constants.DetectorDelta), (int)(imageHeight / constants.DetectorDelta));
            foreach (int dy in Enumerable.Range(0, Detectors.Dimensions[1]))
                foreach (int dx in Enumerable.Range(0, Detectors.Dimensions[0]))
                {
                    Detector detector = new()
                    {
                        X = dx,
                        Y = dy,
                        CenterX = dx * constants.DetectorDelta,
                        CenterY = dy * constants.DetectorDelta,
                    };
                    Detectors[dx, dy] = detector;
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
        public void GenerateOwnedData(Random random, IConstants constants, GradientDistribution gradientDistribution)
        {
            // TODO gradientDistribution -> DetectorRanges            
            int gradientMagnitudeRange = constants.GeneratedMaxGradientMagnitude / constants.MagnitudeRangesCount;

            DetectorsRanges = new DenseMatrix<DetectorRange>(constants.GeneratedMaxGradientMagnitude + gradientMagnitudeRange, 360);                                  

            float gmIn1 = (constants.GeneratedMaxGradientMagnitude - constants.GeneratedMinGradientMagnitude) / MathF.Sqrt(constants.SubAreaMiniColumnsCount!.Value / MathF.PI);

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
                //    angleRange = angleRange0;

                angleRange = MathF.Atan2(constants.K5, gradientMagnitude / gmIn1) * 8.0f;

                if (angleRange > 2 * MathF.PI)
                    angleRange = 2 * MathF.PI;

                foreach (int gradientAngleDegreeIdx in Enumerable.Range(0, DetectorsRanges.Dimensions[1]))
                {
                    DetectorsRanges[gradientMagnitudeIdx, gradientAngleDegreeIdx] = new DetectorRange
                    {
                        GradientMagnitudeRange = gradientMagnitudeRange,
                        GradientAngleRange = angleRange
                    };
                }
            }

            // Плотность детекторов, в зависимости от модуля градиента и угла градиента (в градусах).
            MatrixFloat detectorDensities = new MatrixFloat(DetectorsRanges.Dimensions[0], DetectorsRanges.Dimensions[1]);

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

            MatrixFloat detectorDensities_Accumulative = new MatrixFloat(detectorDensities.Dimensions[0], detectorDensities.Dimensions[1]);
            detectorDensities_Accumulative.Data = DistributionHelper.GetAccumulativeDistribution(detectorDensities.Data);

            //DataToDisplayHolder dataToDisplayHolder = Program.Host.Services.GetRequiredService<DataToDisplayHolder>();
            //dataToDisplayHolder.Distribution = new ulong[360];

            foreach (int di in Enumerable.Range(0, Detectors.Data.Length))
            {
                int rawIndex = DistributionHelper.GetRandom(random, detectorDensities_Accumulative.Data);
                //var indices = detectorDensities_Accumulative.GetIndices(rawIndex);                
                var indices = (random.Next(constants.GeneratedMaxGradientMagnitude + gradientMagnitudeRange), random.Next(360));

                Detector detector = Detectors.Data[di];

                detector.GradientMagnitudeMax = indices.Item1;                
                detector.GradientAngleMax = (float)MathHelper.DegreesToRadians(indices.Item2);                
                detector.BitIndexInHash = random.Next(constants.HashLength);
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
                foreach (int di in Enumerable.Range(0, Detectors.Data.Length))
                {
                    Detector detector = Detectors.Data[di];

                    writer.Write(detector.GradientMagnitudeMax);                    
                    writer.Write(detector.GradientAngleMax);                    
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
                        foreach (int di in Enumerable.Range(0, Detectors.Data.Length))
                        {
                            Detector detector = Detectors.Data[di];

                            detector.GradientMagnitudeMax = reader.ReadSingle();                            
                            detector.GradientAngleMax = reader.ReadSingle();                            
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
        public int X;

        public int Y;        

        /// <summary>
        ///     [0..MNISTImageWidth]
        /// </summary>
        public double CenterX { get; init; }

        /// <summary>
        ///     [0..MNISTImageHeight]
        /// </summary>
        public double CenterY { get; init; }
        
        public float GradientMagnitudeMax;        

        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public float GradientAngleMax;        

        public int BitIndexInHash;

        public bool Temp_IsActivated;

        public GradientInPoint Temp_GradientInPoint;

        public void CalculateIsActivated(Retina retina, DenseMatrix<GradientInPoint> gradientMatrix, IConstants constants, Vector2 offset = default)
        {
            Temp_GradientInPoint = MathHelper.GetInterpolatedGradient(CenterX - offset.X, CenterY - offset.Y, gradientMatrix);

            if (Temp_GradientInPoint.Magnitude < constants.DetectorMinGradientMagnitude)
            {
                Temp_IsActivated = false;
                return;
            }

            int gradientAngleDegree = (int)MathHelper.RadiansToDegrees((float)Temp_GradientInPoint.Angle);

            // TESTCODE
            //DetectorRange detectorRange = retina.DetectorsRanges[(int)Temp_GradientInPoint.Magnitude, gradientAngleDegree];
            DetectorRange detectorRange0 = retina.DetectorsRanges[(int)GradientMagnitudeMax, 0];
            float gradientMagnitudeMin = GradientMagnitudeMax - detectorRange0.GradientMagnitudeRange;
            if (gradientMagnitudeMin < 0)
                gradientMagnitudeMin = 0;
            DetectorRange detectorRange1 = retina.DetectorsRanges[(int)((GradientMagnitudeMax + gradientMagnitudeMin) / 2.0f), 0];            

            bool activated = (Temp_GradientInPoint.Magnitude >= gradientMagnitudeMin) && (Temp_GradientInPoint.Magnitude < GradientMagnitudeMax);
            if (!activated)
            {
                Temp_IsActivated = false;
                return;
            }

            // [-pi, pi)
            float gradientAngleMin = MathHelper.NormalizeAngle(GradientAngleMax - detectorRange1.GradientAngleRange);            
                
            if (GradientAngleMax > gradientAngleMin + 0.01f)
                activated = (Temp_GradientInPoint.Angle >= gradientAngleMin) && (Temp_GradientInPoint.Angle < GradientAngleMax);
            else
                activated = (Temp_GradientInPoint.Angle >= gradientAngleMin) || (Temp_GradientInPoint.Angle < GradientAngleMax);
            Temp_IsActivated = activated;
        }

        public bool GetIsActivated_Obsolete(GradientInPoint[,] gradientMatrix, IConstants constants, Vector2 offset = default)
        {
            (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(CenterX - offset.X, CenterY - offset.Y, gradientMatrix);

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
        ///     Диапазоны модуля градиента.
        /// </summary>
        public float GradientMagnitudeRange;

        /// <summary>
        ///     Диапазоны угла градиента.
        /// </summary>
        public float GradientAngleRange;

        public void SerializeOwnedData(SerializationWriter writer, object? context)
        {
            writer.Write(GradientMagnitudeRange);
            writer.Write(GradientAngleRange);
        }

        public void DeserializeOwnedData(SerializationReader reader, object? context)
        {
            GradientMagnitudeRange = reader.ReadSingle();
            GradientAngleRange = reader.ReadSingle();
        }
    }
}

using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using Ssz.Utils.Serialization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Numerics;
using static Ssz.AI.Models.Cortex;

namespace Ssz.AI.Models
{
    public class Retina : ISerializableModelObject
    {
        #region construction and destruction

        public Retina(ICortexConstants constants, int imageWidth, int imageHeight)
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
        ///     Generates model data after construction.
        /// </summary>
        public void GenerateOwnedData(Random random, ICortexConstants constants, GradientDistribution gradientDistribution)
        {
            UInt64[] magnitudeAccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
            
            //UInt64[] angleAccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.AngleData);
            //ulong maxCount = ulong.MinValue;
            //int maxMagnitude = -1;
            //for (int i = gradientDistribution.MagnitudeData.Length - 1; i >= 0; i -= 1)
            //{
            //    ulong count = gradientDistribution.MagnitudeData[i];
            //    if (count > maxCount)
            //    {
            //        maxCount = count;
            //        maxMagnitude = i;
            //    }
            //}

            foreach (int di in Enumerable.Range(0, Detectors.Data.Length))
            {
                var (gradientMagnitudeLowLimitIndex, gradientMagnitudeHighLimitIndex) = DistributionHelper.GetLimitsIndices(magnitudeAccumulativeDistribution, random, constants.MagnitudeRangesCount);

                //int angleRangeDegree = constants.AngleRangeDegreeMax - (constants.AngleRangeDegreeMax - constants.AngleRangeDegreeMin) * gradientMagnitudeHighLimitIndex / maxMagnitude;
                double gradientMagnitudeHighLimitForAngle = gradientMagnitudeHighLimitIndex;
                double hiLimit = constants.AngleRangeDegreeMinMagnitude;
                if (gradientMagnitudeHighLimitForAngle > hiLimit)
                    gradientMagnitudeHighLimitForAngle = hiLimit;
                double angleRangeDegree = constants.AngleRangeDegreeMax - (constants.AngleRangeDegreeMax - constants.AngleRangeDegreeMin) * gradientMagnitudeHighLimitForAngle / hiLimit;

                double gradientAngleLowLimit = 2 * Math.PI * random.NextDouble() - Math.PI;
                double gradientAngleHighLimit = gradientAngleLowLimit + 2 * Math.PI * angleRangeDegree / 360;
                if (gradientAngleHighLimit > Math.PI)
                    gradientAngleHighLimit = gradientAngleHighLimit - 2 * Math.PI;

                Detector detector = Detectors.Data[di];

                detector.GradientMagnitudeLowLimit = gradientMagnitudeLowLimitIndex;
                detector.GradientMagnitudeHighLimit = gradientMagnitudeHighLimitIndex;
                detector.GradientAngleLowLimit = gradientAngleLowLimit;
                detector.GradientAngleHighLimit = gradientAngleHighLimit;
                detector.BitIndexInHash = random.Next(constants.HashLength);
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
            using (writer.EnterBlock(2))
            {
                foreach (int di in Enumerable.Range(0, Detectors.Data.Length))
                {
                    Detector detector = Detectors.Data[di];

                    writer.Write(detector.GradientMagnitudeLowLimit);
                    writer.Write(detector.GradientMagnitudeHighLimit);
                    writer.Write(detector.GradientAngleLowLimit);
                    writer.Write(detector.GradientAngleHighLimit);
                    writer.Write(detector.BitIndexInHash);
                }
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

                            detector.GradientMagnitudeLowLimit = reader.ReadDouble();
                            detector.GradientMagnitudeHighLimit = reader.ReadDouble();
                            detector.GradientAngleLowLimit = reader.ReadDouble();
                            detector.GradientAngleHighLimit = reader.ReadDouble();
                            detector.BitIndexInHash = reader.ReadOptimizedInt32();
                        }
                        break;
                    case 2:
                        foreach (int di in Enumerable.Range(0, Detectors.Data.Length))
                        {
                            Detector detector = Detectors.Data[di];

                            detector.GradientMagnitudeLowLimit = reader.ReadDouble();
                            detector.GradientMagnitudeHighLimit = reader.ReadDouble();
                            detector.GradientAngleLowLimit = reader.ReadDouble();
                            detector.GradientAngleHighLimit = reader.ReadDouble();
                            detector.BitIndexInHash = reader.ReadInt32();
                        }
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
        ///     Минимальная чувствительность к модулю градиента
        /// </summary>
        public const double GradientMagnitudeMinimum = 5.0;

        /// <summary>
        ///     [0..MNISTImageWidth]
        /// </summary>
        public double CenterX { get; init; }

        /// <summary>
        ///     [0..MNISTImageHeight]
        /// </summary>
        public double CenterY { get; init; }
        
        public double GradientMagnitudeLowLimit;

        public double GradientMagnitudeHighLimit;

        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double GradientAngleLowLimit;

        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double GradientAngleHighLimit;

        public int BitIndexInHash;

        public bool Temp_IsActivated;

        public GradientInPoint Temp_GradientInPoint;

        public void CalculateIsActivated(DenseMatrix<GradientInPoint> gradientMatrix, Vector2 offset = default)
        {
            Temp_GradientInPoint = MathHelper.GetInterpolatedGradient(CenterX - offset.X, CenterY - offset.Y, gradientMatrix);

            if (Temp_GradientInPoint.Magnitude < GradientMagnitudeMinimum)
            {
                Temp_IsActivated = false;
                return;
            }

            bool activated = (Temp_GradientInPoint.Magnitude >= GradientMagnitudeLowLimit) && (Temp_GradientInPoint.Magnitude < GradientMagnitudeHighLimit);
            if (!activated)
            {
                Temp_IsActivated = false;
                return;
            }

            if (GradientAngleHighLimit > GradientAngleLowLimit)
                activated = (Temp_GradientInPoint.Angle >= GradientAngleLowLimit) && (Temp_GradientInPoint.Angle < GradientAngleHighLimit);
            else
                activated = (Temp_GradientInPoint.Angle >= GradientAngleLowLimit) || (Temp_GradientInPoint.Angle < GradientAngleHighLimit);
            Temp_IsActivated = activated;
        }

        public bool GetIsActivated_Obsolete(GradientInPoint[,] gradientMatrix, Vector2 offset = default)
        {
            (double magnitude, double angle) = MathHelper.GetInterpolatedGradient_Obsolete(CenterX - offset.X, CenterY - offset.Y, gradientMatrix);

            if (magnitude < GradientMagnitudeMinimum)
                return false;

            bool activated = (magnitude >= GradientMagnitudeLowLimit) && (magnitude < GradientMagnitudeHighLimit);
            if (!activated)
                return false;

            if (GradientAngleHighLimit > GradientAngleLowLimit)
                activated = (angle >= GradientAngleLowLimit) && (angle < GradientAngleHighLimit);
            else
                activated = (angle >= GradientAngleLowLimit) || (angle < GradientAngleHighLimit);
            return activated;
        }        
    }    
}

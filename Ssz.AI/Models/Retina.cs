using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models
{
    public class Retina
    {        
        public Retina(ICortexConstants constants, GradientDistribution gradientDistribution, int angleRangesCount, int magnitudeRangesCount, int hashLength)
        {
            UInt64[] magnitudeAccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
            UInt64[] angleAccumulativeDistribution = DistributionHelper.GetAccumulativeDistribution(gradientDistribution.AngleData);            

            Random random = new();

            Detectors = new Detector[(int)((MNISTHelper.MNISTImageWidth - 1) / constants.DetectorDelta), (int)((MNISTHelper.MNISTImageHeight - 1) / constants.DetectorDelta)];
            foreach (int dy in Enumerable.Range(0, Detectors.GetLength(1)))
                foreach (int dx in Enumerable.Range(0, Detectors.GetLength(0)))
                {
                    var (gradientMagnitudeLowLimitIndex, gradientMagnitudeHighLimitIndex) = DistributionHelper.GetLimitsIndices(magnitudeAccumulativeDistribution, random, magnitudeRangesCount);                       

                    double gradientAngleLowLimit = 2 * Math.PI * random.NextDouble() - Math.PI;
                    double gradientAngleHighLimit = gradientAngleLowLimit + 2 * Math.PI / angleRangesCount;
                    if (gradientAngleHighLimit > Math.PI)
                        gradientAngleHighLimit = gradientAngleHighLimit - 2 * Math.PI;

                    Detector detector = new()
                    {
                        CenterX = dx * constants.DetectorDelta,
                        CenterY = dy * constants.DetectorDelta,
                        GradientMagnitudeLowLimit = gradientMagnitudeLowLimitIndex,
                        GradientMagnitudeHighLimit = gradientMagnitudeHighLimitIndex,
                        GradientAngleLowLimit = gradientAngleLowLimit,
                        GradientAngleHighLimit = gradientAngleHighLimit,
                        BitIndexInHash = random.Next(hashLength)
                    };
                    Detectors[dx, dy] = detector;
                }                        
        }

        public readonly Detector[,] Detectors;
    }

    public class Detector
    {
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
        //public double Width { get; init; }
        public double GradientMagnitudeLowLimit { get; init; }
        public double GradientMagnitudeHighLimit { get; init; }
        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double GradientAngleLowLimit { get; init; }
        /// <summary>
        ///     [-pi, pi]
        /// </summary>
        public double GradientAngleHighLimit { get; init; }

        public int BitIndexInHash;

        public bool GetIsActivated(GradientInPoint[,] gradientMatrix)
        {
            (double magnitude, double angle) = MathHelper.GetInterpolatedGradient(CenterX, CenterY, gradientMatrix);

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

        public bool Temp_IsActivated;
    }
}

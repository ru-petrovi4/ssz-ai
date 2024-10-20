using Ssz.AI.Grafana;
using Ssz.AI.Helpers;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ssz.AI.Models
{
    public static class DetectorsGenerator
    {        
        public static List<Detector> Generate(GradientDistribution gradientDistribution, int angleRangesCount, int magnitudeRangesCount, int hashLength)
        {
            UInt64[] magnitudeAccumulativeDistribution = GetAccumulativeDistribution(gradientDistribution.MagnitudeData);
            UInt64[] angleAccumulativeDistribution = GetAccumulativeDistribution(gradientDistribution.AngleData);

            Random random = new();
            List<Detector> detectors = new(MNISTHelper.MNISTImageWidth * MNISTHelper.MNISTImageHeight * 100);
            foreach (int i in Enumerable.Range(0, (MNISTHelper.MNISTImageWidth - 1) * 10))
            {
                foreach (int j in Enumerable.Range(0, (MNISTHelper.MNISTImageHeight - 1) * 10))
                {
                    var (gradientMagnitudeLowLimitIndex, gradientMagnitudeHighLimitIndex) = GetLimitsIndices(magnitudeAccumulativeDistribution, random, magnitudeRangesCount);                       

                    double gradientAngleLowLimit = 2 * Math.PI * random.NextDouble() - Math.PI;
                    double gradientAngleHighLimit = gradientAngleLowLimit + 2 * Math.PI / angleRangesCount;
                    if (gradientAngleHighLimit > Math.PI)
                        gradientAngleHighLimit = gradientAngleHighLimit - 2 * Math.PI;

                    Detector detector = new()
                    {
                        CenterX = i / 10.0,
                        CenterY = j / 10.0,
                        GradientMagnitudeLowLimit = gradientMagnitudeLowLimitIndex,
                        GradientMagnitudeHighLimit = gradientMagnitudeHighLimitIndex,
                        GradientAngleLowLimit = gradientAngleLowLimit,
                        GradientAngleHighLimit = gradientAngleHighLimit,
                        BitIndexInHash = random.Next(hashLength)
                    };
                    detectors.Add(detector);
                }
            }
            return detectors;
        }

        private static UInt64[] GetAccumulativeDistribution(UInt64[] distribution)
        {
            UInt64[] result = new UInt64[distribution.Length];
            UInt64 value = 0;
            foreach (int i in Enumerable.Range(0, distribution.Length))
            {
                value += distribution[i];
                result[i] = value;
            }
            return result;
        }

        /// <summary>
        ///     Returns highLimitIndex > lowLimitIndex
        /// </summary>
        /// <param name="accumulativeDistribution"></param>
        /// <param name="random"></param>
        /// <param name="rangesCount"></param>
        /// <returns></returns>
        private static (int lowLimitIndex, int highLimitIndex) GetLimitsIndices(UInt64[] accumulativeDistribution, Random random, int rangesCount)
        {
            UInt64 maxSamples = accumulativeDistribution[^1]; // Последний элемент массиваж
            UInt64 rangeSamples = maxSamples / (UInt64)rangesCount;
            UInt64 lowLimitSamples = (UInt64)(random.NextDouble() * (maxSamples + rangeSamples)) - rangeSamples;
            UInt64 hightLimitSamples = lowLimitSamples + rangeSamples;
            int lowLimitIndex = 0;
            foreach (int i in Enumerable.Range(0, accumulativeDistribution.Length))
            {
                if (lowLimitSamples <= accumulativeDistribution[i])
                {
                    lowLimitIndex = i;
                    break;
                }
            }
            int highLimitIndex = accumulativeDistribution.Length;
            foreach (int i in Enumerable.Range(lowLimitIndex + 1, accumulativeDistribution.Length - lowLimitIndex - 1))
            {
                if (hightLimitSamples <= accumulativeDistribution[i])
                {
                    highLimitIndex = i;
                    break;
                }
            }
            return (lowLimitIndex, highLimitIndex);
        }
    }
}
